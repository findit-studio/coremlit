use super::*;

use core::num::NonZeroU32;

use asry::{
  emissions::{
    EmissionsFailure, EncoderOutput, EnglishNormalizer, OovKind, default_oov_policy,
    wildcard_all_policy,
  },
  time::ANALYSIS_TIMEBASE,
};

use crate::audio::align::error::TokenizationError;

use crate::audio::align::acoustic::{
  AcousticGeometry, Granularity, LetterCase, OutputKind, Tokenization, WordDelimiter,
};

/// A contract of a model's own with the staged tokenization (`|`, upper case,
/// one character at a time, no specials) and log-probability output: only
/// `blank` and `geometry` vary here.
fn contract(blank: u32, geometry: AcousticGeometry) -> AcousticContract {
  AcousticContract::new(
    blank,
    geometry,
    Tokenization::new(
      WordDelimiter::Pipe,
      LetterCase::Upper,
      Granularity::Character,
      &[],
    ),
    OutputKind::LogProbabilities,
  )
}

fn normalizer() -> DynTextNormalizer {
  Box::new(EnglishNormalizer::new())
}

/// The bundled seam: the staged table under the staged contract.
fn bundled_seam() -> EmissionsAligner {
  build_seam(
    Lang::En,
    &Vocabulary::bundled(),
    &AcousticContract::BASE960H,
    normalizer(),
    &AlignerOptions::new(),
  )
  .expect("bundled tokenizer + explicit blank id builds")
}

/// What `event` is and where: its kind, its character and word indices, and
/// its language. An event is made only by detection, so a law compares these.
fn position(event: &OovEvent) -> (OovKind, usize, usize, Lang) {
  (
    event.kind().clone(),
    event.char_index(),
    event.word_index(),
    event.language().clone(),
  )
}

/// The positions of `events`, in order.
fn positions(events: &[OovEvent]) -> Vec<(OovKind, usize, usize, Lang)> {
  events.iter().map(position).collect()
}

/// The clock of a chunk at the stream's start, in the analysis timebase.
fn clock() -> OutputClock {
  OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock")
}

// ---------------------------------------------------------------------
// AlignerOptions (rust-options-pattern)
// ---------------------------------------------------------------------

#[test]
fn options_new_matches_documented_defaults() {
  let o = AlignerOptions::new();
  assert_eq!(o.min_speech_coverage(), DEFAULT_MIN_SPEECH_COVERAGE);
  assert_eq!(o.min_speech_coverage(), 0.5);
  assert_eq!(o.max_intra_silent_run(), DEFAULT_MAX_INTRA_SILENT_RUN);
  // The shipping placement is CpuOnly, and it is a correctness requirement:
  // the ANE placements underflow this model's fp16 `log(softmax(·))` tail to a
  // `-45440` sentinel. See `DEFAULT_ENCODER_COMPUTE`.
  assert_eq!(o.compute(), DEFAULT_ENCODER_COMPUTE);
  assert_eq!(o.compute(), ComputeUnits::CpuOnly);
}

#[test]
fn options_compute_overrides() {
  // Override with placements that are NOT the default, or this would pass
  // against a no-op `with_compute`.
  let o = AlignerOptions::new().with_compute(ComputeUnits::CpuAndGpu);
  assert_eq!(o.compute(), ComputeUnits::CpuAndGpu);

  let mut o = AlignerOptions::new();
  o.set_compute(ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(o.compute(), ComputeUnits::CpuAndNeuralEngine);
}

#[test]
fn options_default_matches_new() {
  assert_eq!(AlignerOptions::default(), AlignerOptions::new());
}

#[test]
fn options_with_builders_override() {
  let o = AlignerOptions::new()
    .with_min_speech_coverage(0.75)
    .with_max_intra_silent_run(Duration::from_millis(120));
  assert_eq!(o.min_speech_coverage(), 0.75);
  assert_eq!(o.max_intra_silent_run(), Duration::from_millis(120));
}

#[test]
fn options_set_in_place() {
  let mut o = AlignerOptions::new();
  o.set_min_speech_coverage(0.25);
  o.set_max_intra_silent_run(Duration::from_millis(40));
  assert_eq!(o.min_speech_coverage(), 0.25);
  assert_eq!(o.max_intra_silent_run(), Duration::from_millis(40));
}

/// [`AlignerOptions`]'s `Display`, pinned byte-exactly: this is the spelling a
/// downstream derivation fingerprint persists, so a silent respelling here
/// would silently invalidate every fingerprint built on it (see the doc on
/// `impl Display for AlignerOptions`).
///
/// Composes [`ComputeUnits`]'s OWN `Display` verbatim rather than re-deriving
/// it — that type is the one place its spelling can change, and a drift there
/// fails this test exactly as readily as one introduced here — and renders
/// `max_intra_silent_run` through `humantime::format_duration`, the same
/// grammar this workspace already uses for a persisted `Duration`.
#[test]
fn aligner_options_display_pins_the_composed_spelling() {
  // Default: coverage 0.5, an 80 ms silent-run tolerance, CpuOnly placement.
  assert_eq!(
    AlignerOptions::new().to_string(),
    "min_speech_coverage=0.5,max_intra_silent_run=80ms,compute=cpu_only"
  );

  // Every field distinct from its default.
  let built = AlignerOptions::new()
    .with_min_speech_coverage(0.75)
    .with_max_intra_silent_run(Duration::from_millis(120))
    .with_compute(ComputeUnits::CpuAndGpu);
  assert_eq!(
    built.to_string(),
    "min_speech_coverage=0.75,max_intra_silent_run=120ms,compute=cpu_and_gpu"
  );
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_missing_fields_default() {
  let o: AlignerOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(o, AlignerOptions::new());
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_partial_fills_defaults() {
  let o: AlignerOptions = serde_json::from_str(r#"{"min_speech_coverage":0.7}"#).unwrap();
  assert_eq!(o.min_speech_coverage(), 0.7);
  assert_eq!(o.max_intra_silent_run(), DEFAULT_MAX_INTRA_SILENT_RUN);
  assert_eq!(o.compute(), DEFAULT_ENCODER_COMPUTE);
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_round_trips() {
  // A non-default compute, so the round-trip proves the field actually
  // survives serialization rather than being re-defaulted on the way back.
  let o = AlignerOptions::new()
    .with_max_intra_silent_run(Duration::from_millis(120))
    .with_compute(ComputeUnits::CpuAndGpu);
  let json = serde_json::to_string(&o).unwrap();
  assert!(json.contains("cpu_and_gpu"), "round-tripped json: {json}");
  let back: AlignerOptions = serde_json::from_str(&json).unwrap();
  assert_eq!(o, back);
}

// ---------------------------------------------------------------------
// Seam construction / blank-id wiring (DECISION 5) — hermetic: these
// build the asry seam alone (bundled tokenizer bytes + a normalizer), no
// CoreML model, so they run without ALIGNKIT_TEST_MODELS.
// ---------------------------------------------------------------------

#[test]
fn build_seam_wires_the_staged_blank_and_vocab_29() {
  let seam = build_seam(
    Lang::En,
    &Vocabulary::bundled(),
    &AcousticContract::BASE960H,
    normalizer(),
    &AlignerOptions::new(),
  )
  .expect("bundled tokenizer + explicit blank id builds");
  assert_eq!(seam.blank_token_id(), AcousticContract::BASE960H.blank());
  assert_eq!(seam.blank_token_id(), 0);
  assert_eq!(
    seam.vocab_size().get(),
    crate::audio::align::vocab::VOCAB_SIZE
  );
}

#[test]
fn build_seam_threads_options_into_the_seam() {
  let options = AlignerOptions::new().with_max_intra_silent_run(Duration::from_millis(120));
  let seam = build_seam(
    Lang::En,
    &Vocabulary::bundled(),
    &AcousticContract::BASE960H,
    normalizer(),
    &options,
  )
  .expect("builds");
  assert_eq!(seam.max_intra_silent_run(), options.max_intra_silent_run());
}

/// A geometry at 16 kHz.
fn geometry(receptive_field: u32, stride: u32) -> AcousticGeometry {
  AcousticGeometry::new(
    16_000,
    NonZeroU32::new(receptive_field).expect("nonzero"),
    NonZeroU32::new(stride).expect("nonzero"),
  )
  .expect("a geometry asry's seam times")
}

#[test]
fn seam_stride_is_the_contract_stride() {
  // THE one-stride invariant. The stride the encoder TRUNCATES by (the
  // contract geometry's, in `encode::truncated_frame_count`) is what times the
  // words: it fixes `T`, and asry maps boundaries by `T` frames over the chunk's
  // real samples. The seam must be HANDED that same number — not because a
  // seam-only mismatch re-times anything (it does not; the grid follows the
  // encoder's `T`, not the hop), but because it would otherwise DECLARE a
  // stride the encoder never used, and asry does not reconcile the two: its
  // frame-count check is a band per hop, which on jfk.wav accepts 321 as well
  // as 320.
  //
  // It holds by construction: `build_seam` reads the stride off the same
  // contract the encoder truncates by. A mutant that re-spells the seam's
  // stride as the staged 320 fails the third case.
  for (contract, stride) in [
    (AcousticContract::BASE960H, 320),
    (contract(0, geometry(640, 320)), 320),
    (contract(0, geometry(480, 480)), 480),
  ] {
    let seam = build_seam(
      Lang::En,
      &Vocabulary::bundled(),
      &contract,
      normalizer(),
      &AlignerOptions::new(),
    )
    .expect("builds");
    assert_eq!(seam.hop_samples(), contract.geometry().stride());
    assert_eq!(
      seam.hop_samples().get(),
      stride,
      "the seam's hop must equal the contract's stride (the stride that times the words, via T)"
    );
  }
}

/// A character the bundled table cannot spell arrives as an OOV EVENT through
/// the seam every [`Aligner`] holds — never as a tokenization failure.
///
/// The bundled table declares `unk_token` `"<unk>"` and holds no such entry, so
/// asry 0.1's per-character `Tokenizer::encode` probe raised `MissingUnkToken`
/// on every character outside its 29 entries and the whole chunk failed before
/// any OOV policy could decide it (`Tokenization: encode('é') failed`). asry
/// asks the vocabulary instead (since 0.2, asry#21), so `é` (the normalizer
/// keeps diacritics), `&` and `4` are three `Symbol` events at their positions
/// in the normalized text.
#[test]
fn a_character_the_bundled_table_cannot_spell_is_an_oov_event() {
  let detection = bundled_seam()
    .detect_oov("Café AT&T b4d")
    .expect("a character the vocabulary cannot spell is an event, never an error");
  assert_eq!(
    positions(detection.events()),
    [
      (OovKind::Symbol('é'), 3, 0, Lang::En),
      (OovKind::Symbol('&'), 7, 1, Lang::En),
      (OovKind::Symbol('4'), 11, 2, Lang::En),
    ]
  );
}

/// **A seam built from a contract reads back exactly the contract's
/// statements**: the word delimiter, the letter case, the receptive field, the
/// stride and the blank, each the contract's and none asry's default. The
/// staged contract states asry's English wav2vec2 defaults (`|`, upper case,
/// 400 samples, a 320-sample hop), so it cannot tell a statement from a
/// default; the second contract differs from every default — no delimiter, as
/// written, a 640-sample field, a 480-sample hop, and a blank at id 1 where
/// asry's guess by name would take the `<pad>` at 0.
///
/// Plant: dropping `.word_delimiter(..)` from `build_seam` reads back asry's
/// `|` for the contract that states none, and this test fails; so does
/// dropping any other statement.
#[test]
fn a_seam_reads_back_the_contracts_statements() {
  let staged = bundled_seam();
  assert_eq!(staged.word_delimiter(), "|");
  assert_eq!(staged.letter_case(), asry::emissions::LetterCase::Upper);
  assert_eq!(staged.receptive_field_samples().get(), 400);
  assert_eq!(staged.hop_samples().get(), 320);
  assert_eq!(staged.blank_token_id(), AcousticContract::BASE960H.blank());

  let vocabulary = table(&["<pad>", "-", "a", "b", "你"]);
  let contract = AcousticContract::new(
    1,
    geometry(640, 480),
    Tokenization::new(
      WordDelimiter::Absent,
      LetterCase::AsWritten,
      Granularity::Character,
      &["<pad>"],
    ),
    OutputKind::Logits,
  );
  assert_eq!(
    check_tokenization(1, contract.tokenization(), &vocabulary, false),
    Ok(()),
    "the contract is one the table and a non-delimiting normalizer agree with"
  );
  let seam = build_seam(
    Lang::Zh,
    &vocabulary,
    &contract,
    Box::new(asry::emissions::ChineseNormalizer::new()),
    &AlignerOptions::new(),
  )
  .expect("builds");
  assert_eq!(
    seam.word_delimiter(),
    "",
    "the contract states no delimiter"
  );
  assert_eq!(seam.letter_case(), asry::emissions::LetterCase::AsWritten);
  assert_eq!(seam.receptive_field_samples().get(), 640);
  assert_eq!(seam.hop_samples().get(), 480);
  assert_eq!(seam.blank_token_id(), 1);
}

/// **asry pads a short chunk to the contract's receptive field**, the one the
/// seam reads back and the encoder truncates by: a 100-sample chunk comes back
/// 400 samples long under the staged contract and 640 under a 640-sample
/// field, and a chunk at least that long comes back unpadded.
#[test]
fn asry_pads_a_short_chunk_to_the_contracts_receptive_field() {
  let abort = AtomicBool::new(false);
  for (contract, cases) in [
    (
      AcousticContract::BASE960H,
      [(100usize, 400usize), (400, 400), (500, 500)],
    ),
    (
      contract(0, geometry(640, 320)),
      [(100, 640), (640, 640), (700, 700)],
    ),
  ] {
    let seam = build_seam(
      Lang::En,
      &Vocabulary::bundled(),
      &contract,
      normalizer(),
      &AlignerOptions::new(),
    )
    .expect("builds");
    for (real, padded) in cases {
      let resolution = seam
        .detect_oov("A")
        .expect("detect_oov")
        .decide(wildcard_all_policy);
      let prepared = seam
        .prepare(
          &vec![0.1f32; real],
          &SpeechSpans::all_speech(),
          "A",
          resolution,
          clock(),
          &abort,
        )
        .expect("prepare");
      assert!(!prepared.is_trivial(), "`A` is alignable");
      assert_eq!(
        prepared.encoder_input().len(),
        padded,
        "{real} samples under {contract:?}"
      );
      assert_eq!(prepared.real_samples(), real);
    }
  }
}

/// The words `seam` aligns `text` to over `samples` samples and `frames` frames
/// of uniform emissions, every OOV event wildcarded: the alignment asry makes
/// of the token stream alone, every path as likely as another. It has no path
/// when the stream holds more tokens than there are frames.
fn aligned_words(
  seam: &EmissionsAligner,
  text: &str,
  samples: usize,
  frames: usize,
) -> Result<Vec<String>, EmissionsError> {
  let abort = AtomicBool::new(false);
  let resolution = seam.detect_oov(text)?.decide(wildcard_all_policy);
  let prepared = seam.prepare(
    &vec![0.1f32; samples],
    &SpeechSpans::all_speech(),
    text,
    resolution,
    clock(),
    &abort,
  )?;
  let vocab = seam.vocab_size();
  let uniform = -(vocab.get() as f32).ln();
  let emissions = prepared.encode_with(|_| {
    Ok::<_, EmissionsError>(EncoderOutput::LogProbs {
      frames,
      vocab,
      data: vec![uniform; frames * vocab.get()],
    })
  })?;
  Ok(
    seam
      .finish(prepared, emissions, &abort)?
      .words()
      .iter()
      .map(|word| word.text().to_owned())
      .collect(),
  )
}

/// **No transcript character is spelled onto a reserved column of the staged
/// table** — Codex R6's two cases, answered by asry 0.3 at run time. The
/// table spells its blank `-` and its delimiter `|`, and a character's lookup
/// used to land on them:
///
/// - `A|B` put the delimiter's token inside the word, one word split into two
///   segments under one word index. Now the `|` is not spelled: it is an
///   `OovKind::Symbol` event, the policy's to decide, and wildcarded it leaves
///   one word, `A|B`; the delimiter's column is reached only by the separator
///   tokenization puts between words.
/// - `well-known` (one word under asry 0.3's normalizer) aligned its hyphen to
///   the blank's column. Now the `-` is a mark nobody reads aloud that the
///   vocabulary does not spell: dropped, no event and no target. Its nine
///   letters align as one word in nine frames, where a table spelling `-` as an
///   ordinary class makes ten tokens of it, which nine frames cannot carry.
///
/// The staged document declares both special
/// (`vocab::tests::the_bundled_document_declares_exactly_the_staged_contracts_specials`),
/// and the seam is stated both too: the blank by id, the delimiter as its
/// token. Either reserves them, so this law holds without the document's
/// `special` flags; [`a_declared_one_character_special_is_never_spelled`] is
/// the one the flags alone carry.
#[test]
fn the_staged_seam_spells_no_character_onto_a_reserved_column() {
  let seam = bundled_seam();
  // 16,000 samples make 49 staged frames, 2,960 make 9 (the band admits 9..=10).
  let (second, short) = ((16_000, 49), (2_960, 9));

  let pipe = seam.detect_oov("A|B").expect("detect_oov");
  assert_eq!(
    positions(pipe.events()),
    [(OovKind::Symbol('|'), 1, 0, Lang::En)],
    "the `|` inside a word is no delimiter"
  );
  assert_eq!(
    aligned_words(&seam, "A|B", second.0, second.1).expect("aligns"),
    ["A|B"]
  );

  let hyphen = seam.detect_oov("well-known").expect("detect_oov");
  assert!(hyphen.events().is_empty(), "{:?}", hyphen.events());
  assert_eq!(
    aligned_words(&seam, "well-known", short.0, short.1).expect("nine tokens in nine frames"),
    ["well-known"]
  );

  let lexical_hyphen = build_seam(
    Lang::En,
    &table(&["<pad>", "|", "-", "W", "E", "L", "K", "N", "O", "A", "B"]),
    &contract(0, AcousticGeometry::WAV2VEC2),
    normalizer(),
    &AlignerOptions::new(),
  )
  .expect("builds");
  assert!(
    matches!(
      aligned_words(&lexical_hyphen, "well-known", short.0, short.1),
      Err(EmissionsError::NoAlignmentPath(_))
    ),
    "a `-` the table spells as an ordinary class is a tenth token"
  );
  assert_eq!(
    aligned_words(&lexical_hyphen, "well-known", second.0, second.1).expect("aligns"),
    ["well-known"]
  );
}

/// **A space-delimited table builds a seam whose word delimiter is the space**
/// (Q456): the contract states it — `WordDelimiter::from_token(" ")`, as a
/// model's configuration names it — asry's builder takes it
/// (`word_delimiter(" ")`), and the seam reads it back. The words split at the
/// separator tokenization inserts: `A B` is two words and three tokens, the
/// space's column between the letters, so two frames cannot carry it and three
/// can. The document declares the space special, so a space is never spelled
/// from the text; only the inserted separator reaches its column.
///
/// Plant: `from_token` refusing the space, as it did, fails this law at the
/// contract.
#[test]
fn a_space_delimited_table_builds_a_seam_split_at_the_space() {
  let delimiter = WordDelimiter::from_token(" ").expect("the space is a word delimiter");
  let vocabulary = table(&["<pad>", " ", "A", "B"]);
  let contract = AcousticContract::new(
    0,
    AcousticGeometry::WAV2VEC2,
    Tokenization::new(delimiter, LetterCase::Upper, Granularity::Character, &[]),
    OutputKind::Logits,
  );
  assert_eq!(
    check_tokenization(0, contract.tokenization(), &vocabulary, true),
    Ok(())
  );
  let seam = build_seam(
    Lang::En,
    &vocabulary,
    &contract,
    normalizer(),
    &AlignerOptions::new(),
  )
  .expect("builds");
  assert_eq!(seam.word_delimiter(), " ");

  let detection = seam.detect_oov("A B").expect("detect_oov");
  assert!(detection.events().is_empty(), "{:?}", detection.events());
  // 1,040 samples make three staged frames, 720 make two.
  assert_eq!(
    aligned_words(&seam, "A B", 1_040, 3).expect("three tokens in three frames"),
    ["A", "B"]
  );
  assert!(
    matches!(
      aligned_words(&seam, "A B", 720, 2),
      Err(EmissionsError::NoAlignmentPath(_))
    ),
    "the separator is a token of its own"
  );
}

/// **A one-character special the contract declares is never spelled**, and the
/// document's `special` flag is what reserves it: neither the blank nor the
/// delimiter, `#` is reserved only because the document declares it special
/// added token. Under a contract naming it special, the `#` of `A#B` is an
/// `OovKind::Symbol` event, the policy's to decide; under one that does not,
/// the same table spells it onto its column.
///
/// Plant: writing the document's added tokens without their `special` flag
/// spells the declared `#` onto its column, and this test fails.
#[test]
fn a_declared_one_character_special_is_never_spelled() {
  let vocabulary = table(&["<pad>", "|", "A", "B", "#"]);
  let declared = AcousticContract::new(
    0,
    AcousticGeometry::WAV2VEC2,
    Tokenization::new(
      WordDelimiter::Pipe,
      LetterCase::Upper,
      Granularity::Character,
      &["#"],
    ),
    OutputKind::Logits,
  );
  let seam = |contract: &AcousticContract| {
    build_seam(
      Lang::En,
      &vocabulary,
      contract,
      normalizer(),
      &AlignerOptions::new(),
    )
    .expect("builds")
  };

  let special = seam(&declared).detect_oov("A#B").expect("detect_oov");
  assert_eq!(
    positions(special.events()),
    [(OovKind::Symbol('#'), 1, 0, Lang::En)]
  );

  let ordinary = seam(&contract(0, AcousticGeometry::WAV2VEC2))
    .detect_oov("A#B")
    .expect("detect_oov");
  assert!(
    ordinary.events().is_empty(),
    "an undeclared `#` the table spells is a token: {:?}",
    ordinary.events()
  );
}

// ---------------------------------------------------------------------
// The reserved set, read back. asry reserves the blank and the delimiter it is
// stated, the unknown token the tokenizer document declares and every special
// added token the document still holds once parsed. `build_seam` reads that set
// back (`seam_reserved`) and refuses a seam whose set is not the contract's
// non-lexical one (`check_reserved`). Plant (`check_reserved` accepting any
// set): the refusal law fails.
// ---------------------------------------------------------------------

/// A contract of a model's own: blank 0, wav2vec2's front end, `delimiter`,
/// upper case, one character at a time, `specials` named, raw logits.
fn contract_declaring(
  delimiter: WordDelimiter,
  specials: &'static [&'static str],
) -> AcousticContract {
  AcousticContract::new(
    0,
    AcousticGeometry::WAV2VEC2,
    Tokenization::new(
      delimiter,
      LetterCase::Upper,
      Granularity::Character,
      specials,
    ),
    OutputKind::Logits,
  )
}

/// The columns the seam built from `vocabulary` under `contract` reserves, read
/// back, beside the contract's non-lexical set; the door's tokenization check
/// passes first, as it does before any seam the door builds.
fn reserved_and_declared(
  language: Lang,
  vocabulary: &Vocabulary,
  contract: &AcousticContract,
  normalizer: DynTextNormalizer,
) -> (BTreeSet<usize>, BTreeSet<usize>) {
  assert_eq!(
    check_tokenization(
      contract.blank(),
      contract.tokenization(),
      vocabulary,
      normalizer.use_word_delimiter()
    ),
    Ok(()),
    "the door accepts {contract:?}"
  );
  let seam = build_seam(
    language,
    vocabulary,
    contract,
    normalizer,
    &AlignerOptions::new(),
  )
  .expect("builds");
  let reserved =
    seam_reserved(&seam, &vocabulary.tokenizer_json(contract)).expect("the document parses");
  let declared = vocabulary.non_lexical(contract.blank(), contract.tokenization());
  (reserved, declared)
}

/// **The seam reserves exactly the contract's non-lexical set**, read back from
/// asry's readers and the tokenizer document it parsed: on the staged table
/// under `BASE960H`, `{0, 1}` (`-` and `|`); on a table declaring `#` special,
/// `{0, 1, 4}`; with an empty special at the blank's id, `{0, 1}` — the parse
/// drops the empty entry, and the blank's stated id reserves its column; and
/// under no delimiter with an empty special at id 3, `{0, 3}` — the seam's
/// stated delimiter, the empty token, looks the table's empty entry up.
#[test]
fn the_seam_reserves_exactly_the_contracts_non_lexical_set() {
  let english = || -> DynTextNormalizer { normalizer() };
  let unsegmented = || -> DynTextNormalizer { Box::new(asry::emissions::ChineseNormalizer::new()) };
  let cases = [
    (
      Lang::En,
      Vocabulary::bundled(),
      AcousticContract::BASE960H,
      english(),
      BTreeSet::from([0, 1]),
    ),
    (
      Lang::En,
      table(&["<pad>", "|", "A", "B", "#"]),
      contract_declaring(WordDelimiter::Pipe, &["#"]),
      english(),
      BTreeSet::from([0, 1, 4]),
    ),
    (
      Lang::En,
      table(&["", "|", "A", "B"]),
      contract_declaring(WordDelimiter::Pipe, &[""]),
      english(),
      BTreeSet::from([0, 1]),
    ),
    (
      Lang::Zh,
      table(&["<pad>", "A", "B", ""]),
      contract_declaring(WordDelimiter::Absent, &[""]),
      unsegmented(),
      BTreeSet::from([0, 3]),
    ),
  ];
  for (language, vocabulary, contract, normalizer, expected) in cases {
    let (reserved, declared) = reserved_and_declared(language, &vocabulary, &contract, normalizer);
    assert_eq!(declared, expected, "{contract:?}");
    assert_eq!(reserved, declared, "{contract:?}");
  }
}

/// **A seam that reserves other columns than the contract declares is refused
/// by name once built**, either way round. `build_seam` does not run the door's
/// tokenization check, so a pair the door refuses first reaches asry here:
///
/// - an empty special at id 4, beside a blank at 0 and `|` at 1 (the door's
///   `TokenizationError::EmptySpecial(4)`): the document declares it special,
///   the parse drops it, and the seam reserves `{0, 1}` where the contract
///   declares `{0, 1, 4}`;
/// - HuggingFace's 32-class table under a contract naming none of its specials
///   (the door's `NotCharacterLevel("<s>")`): the document declares `<unk>` the
///   unknown token, so the seam reserves its column 3, which that contract
///   calls lexical — `{0, 3, 4}` where it declares `{0, 4}`.
///
/// Plant: `check_reserved` accepting any set builds both seams, and this law
/// fails.
#[test]
fn a_seam_that_reserves_other_columns_than_declared_is_refused_by_name() {
  let refusal = |vocabulary: &Vocabulary, contract: &AcousticContract| match build_seam(
    Lang::En,
    vocabulary,
    contract,
    normalizer(),
    &AlignerOptions::new(),
  ) {
    Err(AlignerError::ReservedSetMismatch(mismatch)) => {
      (mismatch.declared().to_vec(), mismatch.reserved().to_vec())
    }
    Err(other) => panic!("{contract:?}: refused for another reason: {other}"),
    Ok(_) => panic!("{contract:?}: the seam was built"),
  };

  assert_eq!(
    refusal(
      &table(&["<pad>", "|", "A", "B", ""]),
      &contract_declaring(WordDelimiter::Pipe, &[""])
    ),
    (vec![0, 1, 4], vec![0, 1]),
    "the declared empty special is dropped by the parse"
  );
  let hf = Vocabulary::from_json(HF_BASE960H_TABLE).expect("the table reads");
  assert_eq!(
    refusal(&hf, &contract_declaring(WordDelimiter::Pipe, &[])),
    (vec![0, 4], vec![0, 3, 4]),
    "the declared unknown token is reserved whatever the contract calls it"
  );
}

/// **The door refuses an empty named special by name, before the model
/// loads**: the model path does not exist, so the refusal is the table's and
/// the contract's.
#[test]
fn the_door_refuses_an_empty_named_special_before_the_model_loads() {
  let refused = Aligner::from_paths_with_vocabulary(
    Lang::En,
    Path::new("/nonexistent/model.mlmodelc"),
    &table(&["<pad>", "|", "A", "B", ""]),
    &contract_declaring(WordDelimiter::Pipe, &[""]),
    normalizer(),
    AlignerOptions::new(),
  )
  .err()
  .expect("refused");
  assert_eq!(
    refused,
    AlignerError::Tokenization(TokenizationError::EmptySpecial(4))
  );
}

#[test]
fn bundled_tokenizer_has_no_autodetectable_blank() {
  // Proves the explicit `.blank_token_id(contract.blank())` in `build_seam` is
  // load-bearing: WITHOUT it, asry's default `<pad>` / `[PAD]` / `<blank>`
  // auto-detect finds nothing in the chordai vocab and construction FAILS.
  // A mutant dropping that override would regress to exactly this error.
  let result =
    EmissionsAligner::builder(Lang::En, crate::audio::align::vocab::tokenizer_json_bytes())
      .normalizer(normalizer())
      .build();
  assert!(
    matches!(result, Err(EmissionsError::Config(_))),
    "auto-detect must fail without an explicit blank id"
  );
}

// ---------------------------------------------------------------------
// F3: options() reports EFFECTIVE (post-clamp) state, not the requested value.
// The transform is hermetic (build the seam, read its applied coverage back);
// the wiring through the real `Aligner::options()` is model-gated below.
// ---------------------------------------------------------------------

#[test]
fn effective_options_reports_the_seams_clamped_coverage_not_the_requested_value() {
  // Out-of-range requests are coerced by the seam; effective options must report
  // the coerced value, so `Aligner::options()` never lies about the filter in
  // force. A mutant that stored/returned the requested value fails here.
  for (requested, effective) in [(2.0_f32, 1.0_f32), (-0.25, 0.0)] {
    let options = AlignerOptions::new().with_min_speech_coverage(requested);
    let seam = build_seam(
      Lang::En,
      &Vocabulary::bundled(),
      &AcousticContract::BASE960H,
      normalizer(),
      &options,
    )
    .expect("builds");
    let eff = effective_options(&seam, &options);
    assert_eq!(
      eff.min_speech_coverage(),
      effective,
      "requested {requested} must report as {effective}"
    );
    // ...and it equals the seam's own applied value exactly (the source of truth).
    assert_eq!(eff.min_speech_coverage(), seam.min_speech_coverage().get());
  }

  // NaN → the seam's default, never NaN.
  let options = AlignerOptions::new().with_min_speech_coverage(f32::NAN);
  let seam = build_seam(
    Lang::En,
    &Vocabulary::bundled(),
    &AcousticContract::BASE960H,
    normalizer(),
    &options,
  )
  .expect("builds");
  let eff = effective_options(&seam, &options);
  assert!(
    !eff.min_speech_coverage().is_nan(),
    "NaN must not survive into effective options"
  );
  assert_eq!(eff.min_speech_coverage(), DEFAULT_MIN_SPEECH_COVERAGE);
  assert_eq!(eff.min_speech_coverage(), seam.min_speech_coverage().get());
}

#[test]
fn effective_options_passes_through_the_uncoerced_fields() {
  // Only min_speech_coverage is coerced; max_intra_silent_run and compute pass
  // through from the request untouched. A mutant that rebuilt options from seam
  // defaults (dropping the request) fails here.
  let options = AlignerOptions::new()
    .with_max_intra_silent_run(Duration::from_millis(120))
    .with_compute(ComputeUnits::CpuAndGpu)
    .with_min_speech_coverage(2.0);
  let seam = build_seam(
    Lang::En,
    &Vocabulary::bundled(),
    &AcousticContract::BASE960H,
    normalizer(),
    &options,
  )
  .expect("builds");
  let eff = effective_options(&seam, &options);
  assert_eq!(eff.max_intra_silent_run(), Duration::from_millis(120));
  assert_eq!(eff.compute(), ComputeUnits::CpuAndGpu);
  assert_eq!(eff.min_speech_coverage(), 1.0); // the one field that IS coerced
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn aligner_options_reports_effective_coverage_after_construction() {
  // The wiring proof: a real Aligner built with an out-of-range coverage must
  // report the CLAMPED value through `options()`. Reverting `from_paths_with` to
  // store the requested value fails exactly here (the mutation proof for F3's
  // wiring, which the hermetic `effective_options` tests cannot see).
  let aligner = Aligner::from_paths_with(
    Lang::En,
    &models_dir().join("base960h_aligner.mlmodelc"),
    normalizer(),
    AlignerOptions::new().with_min_speech_coverage(2.0),
  )
  .expect("load base960h_aligner.mlmodelc (set ALIGNKIT_TEST_MODELS)");
  assert_eq!(
    aligner.options().min_speech_coverage(),
    1.0,
    "options() must report the seam's clamped coverage, not the requested 2.0"
  );
}

/// `ALIGNKIT_TEST_MODELS`, or `<workspace>/Models/alignkit` — the crate's
/// convention, duplicated here (as `encode::tests` and `registry::tests` do)
/// because a `src/` unit test cannot import the `tests/` integration crate.
fn models_dir() -> std::path::PathBuf {
  std::env::var_os("ALIGNKIT_TEST_MODELS").map_or_else(
    || crate::tests::models_root().join("alignkit"),
    std::path::PathBuf::from,
  )
}

// ---------------------------------------------------------------------
// The seam's per-chunk outcomes are NAMED — `seam_error`, the classifier both
// seam calls in `align_chunk` go through, tested directly, then through asry's
// real `prepare`.
// ---------------------------------------------------------------------

fn failure(message: &str) -> EmissionsFailure {
  EmissionsFailure::new(message.into())
}

/// `text` detected by the bundled seam and decided by `policy`.
fn decided(text: &str, policy: impl FnMut(&OovEvent) -> OovDecision) -> OovResolution {
  bundled_seam()
    .detect_oov(text)
    .expect("detect_oov")
    .decide(policy)
}

/// **A fail-closed refusal is named, and names every refused position.** The
/// refusal used to come back as an EMPTY result, the same answer as a chunk the
/// lattice cannot align and a chunk with nothing to align. It is
/// `AlignError::Refused` now, carrying the events the caller's decisions
/// resolved `FailClosed` — both of them, in the order detection reported them —
/// and not the one it chose to wildcard.
#[test]
fn seam_error_names_a_refusal_by_every_refused_position() {
  let resolution = decided("Café AT&T b4d", |event| match event.char() {
    Some('é') => OovDecision::Wildcard,
    _ => OovDecision::FailClosed,
  });
  let err = seam_error(
    EmissionsError::SemanticOutOfVocab(failure("OOV '&' resolved as FailClosed")),
    &refused_positions(&resolution),
  );
  let AlignError::Refused(refusal) = err else {
    panic!("a fail-closed refusal must be AlignError::Refused, got {err:?}");
  };
  assert_eq!(
    positions(refusal.events()),
    [
      (OovKind::Symbol('&'), 7, 1, Lang::En),
      (OovKind::Symbol('4'), 11, 2, Lang::En),
    ]
  );
  assert_eq!(refusal.to_string(), "'&' (word 1), '4' (word 2)");
}

/// **An unalignable chunk is the other named case.** `NoAlignmentPath` is
/// `AlignError::NoAlignmentPath` carrying asry's diagnostic — whatever the
/// decisions say, since it is the seam's error that names the case, not the
/// caller's policy (a fail-closed decision is present here on purpose).
#[test]
fn seam_error_names_a_chunk_with_no_alignment_path() {
  let resolution = decided("AT&T", |_| OovDecision::FailClosed);
  let refused = refused_positions(&resolution);
  assert_eq!(refused.len(), 1, "the `&` is refused");
  let err = seam_error(
    EmissionsError::NoAlignmentPath(failure("no finite path")),
    &refused,
  );
  let AlignError::NoAlignmentPath(diagnostic) = err else {
    panic!("a lattice with no path must be AlignError::NoAlignmentPath, got {err:?}");
  };
  assert_eq!(diagnostic.message(), "no finite path");
}

/// asry refuses only at a `FailClosed` decision, so a `SemanticOutOfVocab`
/// with none cannot happen by its contract. If it ever did, the error stays
/// asry's own: a refusal that names no position would be a lie.
#[test]
fn seam_error_never_names_a_refusal_with_no_refused_position() {
  let resolution = decided("b4d", wildcard_all_policy);
  assert_eq!(resolution.resolved().len(), 1, "the `4` is decided");
  let refused = refused_positions(&resolution);
  assert!(refused.is_empty(), "a wildcard refuses nothing");
  assert!(matches!(
    seam_error(
      EmissionsError::SemanticOutOfVocab(failure("fail-closed OOV")),
      &refused
    ),
    AlignError::Alignment(EmissionsError::SemanticOutOfVocab(_))
  ));
}

/// Every other seam failure stays a hard [`AlignError::Alignment`] — the
/// distinction that stops a broken setup from being mistaken for a chunk-level
/// outcome.
#[test]
fn seam_error_passes_every_other_failure_through() {
  assert!(matches!(
    seam_error(EmissionsError::Config(failure("blank id >= V")), &[]),
    AlignError::Alignment(EmissionsError::Config(_))
  ));
  assert!(matches!(
    seam_error(EmissionsError::Aborted(failure("aborted")), &[]),
    AlignError::Alignment(EmissionsError::Aborted(_))
  ));
  assert!(matches!(
    seam_error(
      EmissionsError::Tokenization(failure("stale decisions")),
      &[]
    ),
    AlignError::Alignment(EmissionsError::Tokenization(_))
  ));
}

/// The refusal as it actually arises: asry's own `prepare` over the bundled
/// table, with the caller's policy refusing the `&` that `detect_oov` reported
/// (asry reports it rather than failing on it). `prepare` needs no model — the
/// refusal is decided before the encoder would run — and what reaches the
/// caller is the named refusal of exactly that position. The same text under a
/// policy that wildcards everything prepares a chunk to align.
#[test]
fn a_fail_closed_decision_is_a_named_refusal_through_the_seam() {
  let seam = bundled_seam();
  let text = "Café AT&T b4d";
  let samples = vec![0.0f32; 16_000];
  let abort = AtomicBool::new(false);

  let resolution = seam
    .detect_oov(text)
    .expect("detect_oov")
    .decide(default_oov_policy);
  let refused = refused_positions(&resolution);
  let err = seam
    .prepare(
      &samples,
      &SpeechSpans::all_speech(),
      text,
      resolution,
      clock(),
      &abort,
    )
    .err()
    .map(|err| seam_error(err, &refused))
    .expect("the default policy fails closed on `&`");
  let AlignError::Refused(refusal) = err else {
    panic!("expected the named refusal, got {err:?}");
  };
  assert_eq!(
    positions(refusal.events()),
    [(OovKind::Symbol('&'), 7, 1, Lang::En)]
  );

  let wildcards = seam
    .detect_oov(text)
    .expect("detect_oov")
    .decide(wildcard_all_policy);
  let prepared = seam
    .prepare(
      &samples,
      &SpeechSpans::all_speech(),
      text,
      wildcards,
      clock(),
      &abort,
    )
    .expect("a character the policy wildcards is aligned around, not refused");
  assert!(!prepared.is_trivial());
}

/// **A chunk with tokens and no audio has no alignment path, named before the
/// encoder runs.** asry pads the empty chunk to one receptive field of zeros,
/// and the encoder keeps no frame for no real audio; asry's frame-count check
/// would refuse those zero frames as a stride mismatch, a fault of the model.
/// A text with nothing to align is no such chunk: it is answered as trivial.
#[test]
fn an_empty_chunk_with_tokens_has_no_alignment_path() {
  let seam = bundled_seam();
  let abort = AtomicBool::new(false);
  let prepare = |text: &'static str| {
    let resolution = seam
      .detect_oov(text)
      .expect("detect_oov")
      .decide(wildcard_all_policy);
    seam
      .prepare(
        &[],
        &SpeechSpans::all_speech(),
        text,
        resolution,
        clock(),
        &abort,
      )
      .expect("prepare an empty chunk")
  };
  let prepared = prepare("test");
  assert!(!prepared.is_trivial());
  assert_eq!(prepared.real_samples(), 0);
  assert!(matches!(
    check_audio(&prepared),
    Err(AlignError::NoAlignmentPath(_))
  ));
  assert!(check_audio(&prepare("  ... ")).is_ok(), "nothing to align");
}

// ---------------------------------------------------------------------
// The vocabulary handshake, and a seam built from a table read as JSON.
// ---------------------------------------------------------------------

fn width(width: usize) -> NonZeroUsize {
  NonZeroUsize::new(width).expect("nonzero")
}

/// **A vocabulary of another width is refused by name at load.** The handshake
/// used to be a `debug_assert` against the bundled table's constant, which a
/// release build skipped and which a model of another width never reached (the
/// encoder refused every head but 29). It is the named
/// `AlignerError::VocabularyMismatch` now, carrying both widths, in both
/// directions.
#[test]
fn check_vocabulary_width_refuses_a_table_of_another_width_by_name() {
  assert_eq!(check_vocabulary_width(width(29), width(29)), Ok(()));
  for (vocabulary, model) in [(30, 29), (28, 29), (29, 32)] {
    let Err(AlignerError::VocabularyMismatch(mismatch)) =
      check_vocabulary_width(width(vocabulary), width(model))
    else {
      panic!("a {vocabulary}-entry table on a {model}-class head must be refused by name");
    };
    assert_eq!(
      (mismatch.vocabulary(), mismatch.model()),
      (vocabulary, model)
    );
  }
}

/// The bundled table, read back as the `{token: id}` JSON a model ships beside
/// it (the shape `base960h_dict.json` has, the file it was derived from).
fn bundled_table_as_json() -> Vec<u8> {
  let asset: serde_json::Value =
    serde_json::from_slice(crate::audio::align::vocab::tokenizer_json_bytes())
      .expect("the bundled asset is JSON");
  serde_json::to_vec(&asset["model"]["vocab"]).expect("a table serializes")
}

/// **A table read as JSON builds the seam the bundled document builds** under
/// the staged contract. The same width, the same blank, and — asked about texts
/// that spell whole, hold
/// characters the table cannot spell, carry punctuation, or normalize to
/// nothing — the same OOV events and the same prepared chunks. The model-gated
/// half, `tests/align/align_chunk.rs`, aligns the staged model through its own
/// `base960h_dict.json` and through the bundled table and compares the words.
#[test]
fn a_table_read_as_json_builds_the_bundled_seam() {
  let read = Vocabulary::from_json(&bundled_table_as_json()).expect("the table reads");
  let options = AlignerOptions::new();
  let bundled = build_seam(
    Lang::En,
    &Vocabulary::bundled(),
    &AcousticContract::BASE960H,
    normalizer(),
    &options,
  )
  .expect("builds");
  let own = build_seam(
    Lang::En,
    &read,
    &AcousticContract::BASE960H,
    normalizer(),
    &options,
  )
  .expect("builds");

  assert_eq!(own.vocab_size(), bundled.vocab_size());
  assert_eq!(own.blank_token_id(), bundled.blank_token_id());

  let samples = vec![0.0f32; 16_000];
  let abort = AtomicBool::new(false);
  for text in [
    "And so my fellow Americans, ask not.",
    "Café AT&T b4d",
    "don't stop U.S.A",
    "  ... !! ",
    "1000",
  ] {
    let events =
      |seam: &EmissionsAligner| positions(seam.detect_oov(text).expect("detect_oov").events());
    assert_eq!(events(&own), events(&bundled), "{text:?}");
    // A resolution applies only in the seam that detected it: each seam
    // prepares with its own.
    let prepare = |seam: &EmissionsAligner| {
      let resolution = seam
        .detect_oov(text)
        .expect("detect_oov")
        .decide(wildcard_all_policy);
      seam
        .prepare(
          &samples,
          &SpeechSpans::all_speech(),
          text,
          resolution,
          clock(),
          &abort,
        )
        .map(|prepared| (prepared.is_trivial(), prepared.encoder_input().to_vec()))
    };
    assert_eq!(prepare(&own), prepare(&bundled), "{text:?}");
  }
}

/// HuggingFace's 32-class `wav2vec2-base-960h` table (`vocab.json`).
const HF_BASE960H_TABLE: &[u8] = br#"{"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4,
  "E": 5, "T": 6, "A": 7, "O": 8, "N": 9, "I": 10, "H": 11, "S": 12, "R": 13, "D": 14, "L": 15,
  "U": 16, "M": 17, "W": 18, "C": 19, "F": 20, "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25,
  "K": 26, "'": 27, "X": 28, "J": 29, "Q": 30, "Z": 31}"#;

/// A table of another width builds its own seam under its own contract — the
/// groundwork a per-language aligner stands on. HuggingFace's 32-class
/// `wav2vec2-base-960h` table keeps `<s>`, `</s>` and `<unk>` as entries, and
/// its `config.json` names the blank `pad_token_id: 0`, the `<pad>` entry: its
/// contract states that blank and names the three special, which the door's
/// tokenization check and the seam's reserved set both agree with.
#[test]
fn a_table_of_another_width_builds_a_seam_of_that_width() {
  let vocabulary = Vocabulary::from_json(HF_BASE960H_TABLE).expect("the table reads");
  let contract = contract_declaring(WordDelimiter::Pipe, &["<s>", "</s>", "<unk>"]);
  assert_eq!(
    check_tokenization(0, contract.tokenization(), &vocabulary, true),
    Ok(())
  );
  let seam = build_seam(
    Lang::En,
    &vocabulary,
    &contract,
    normalizer(),
    &AlignerOptions::new(),
  )
  .expect("a 32-class table builds its seam");
  assert_eq!(seam.vocab_size().get(), 32);
  assert_eq!(seam.blank_token_id(), 0);
  assert_eq!(
    positions(seam.detect_oov("b4d").expect("detect_oov").events()),
    [(OovKind::Symbol('4'), 1, 0, Lang::En)]
  );
}

/// **An explicit blank that is not in the table is refused by name.** The
/// contract states the blank; the table's ids run `0..n`, so a blank of `n` or
/// beyond is no column at all, and asry's builder would take it at its word
/// and refuse only in the trellis, on every chunk. Refused here instead, at
/// load, naming the blank and the table's size — every id inside passes,
/// including the last.
#[test]
fn an_explicit_blank_outside_the_table_is_refused_by_name() {
  let entries = width(29);
  for blank in [0u32, 1, 28] {
    assert_eq!(check_blank(blank, entries), Ok(()), "id {blank}");
  }
  for blank in [29u32, 30, u32::MAX] {
    let Err(AlignerError::BlankOutOfVocabulary(refused)) = check_blank(blank, entries) else {
      panic!("id {blank} is no id of a 29-entry table");
    };
    assert_eq!((refused.blank(), refused.entries()), (blank, 29));
  }
}

/// **A table that could be read two ways binds the blank its contract names,
/// and no other.** `{"<blank>": 0, "<pad>": 1, …}` names two conventional
/// blanks; asry's own auto-detect, and the name-priority guess this crate once
/// made, take `<pad>` at id 1 and would read every silence from the wrong
/// column without an error. The table carries no blank of its own now, and
/// there is no door from a table to a seam that does not take a contract: the
/// seam's blank is the contract's id — 0 when the model's blank is `<blank>`,
/// 1 when it is `<pad>` — never a name's.
#[test]
fn an_ambiguous_table_binds_exactly_the_contracts_blank() {
  let table = br#"{"<blank>": 0, "<pad>": 1, "|": 2, "A": 3, "B": 4, "C": 5}"#;
  let vocabulary = Vocabulary::from_json(table).expect("the table reads");
  let document = vocabulary.tokenizer_json(&contract(0, AcousticGeometry::WAV2VEC2));
  let guessed = EmissionsAligner::builder(Lang::En, &document)
    .normalizer(normalizer())
    .build()
    .expect("asry's auto-detect builds a seam");
  assert_eq!(
    guessed.blank_token_id(),
    1,
    "a guess by name takes `<pad>`, the wrong column when the blank is `<blank>`"
  );
  for blank in [0u32, 1] {
    let contract = contract(blank, AcousticGeometry::WAV2VEC2);
    let seam = build_seam(
      Lang::En,
      &vocabulary,
      &contract,
      normalizer(),
      &AlignerOptions::new(),
    )
    .expect("builds");
    assert_eq!(seam.blank_token_id(), blank);
  }
}

// ---------------------------------------------------------------------
// The one composition: this aligner's seam, this aligner's encoder, one
// contract. `Encoder` and `EncoderInput` are crate-private (the `encode`
// module doc's compile_fail doctests pin that), so these laws drive the
// composition from inside the crate, on the staged model.
// ---------------------------------------------------------------------

/// A table from `tokens`, each at its index.
fn table(tokens: &[&str]) -> Vocabulary {
  let entries: Vec<String> = tokens
    .iter()
    .enumerate()
    .map(|(id, token)| format!("{}: {id}", serde_json::to_string(token).expect("a token")))
    .collect();
  Vocabulary::from_json(format!("{{{}}}", entries.join(", ")).as_bytes()).expect("a table")
}

/// **A contradicted tokenization is refused through the public door, before
/// the model loads.** The model path does not exist: the refusal is the
/// table's, the normalizer's and the contract's, decided at load before any
/// model is read — a `|`-containing space-delimited table, and the `A`/`B`/`b`
/// table under both case statements.
#[test]
fn the_door_refuses_a_contradicted_tokenization_before_the_model_loads() {
  let absent = Path::new("/nonexistent/model.mlmodelc");
  let load = |vocabulary: &Vocabulary, contract: &AcousticContract| {
    Aligner::from_paths_with_vocabulary(
      Lang::En,
      absent,
      vocabulary,
      contract,
      normalizer(),
      AlignerOptions::new(),
    )
    .err()
    .expect("refused")
  };
  let staged = contract(0, AcousticGeometry::WAV2VEC2);
  let as_written = AcousticContract::new(
    0,
    AcousticGeometry::WAV2VEC2,
    Tokenization::new(
      WordDelimiter::Pipe,
      LetterCase::AsWritten,
      Granularity::Character,
      &[],
    ),
    OutputKind::LogProbabilities,
  );

  let spaced = table(&["<pad>", " ", "|", "A", "B"]);
  assert_eq!(
    load(&spaced, &staged),
    AlignerError::Tokenization(TokenizationError::WhitespaceToken(" ".to_owned()))
  );
  let mixed = table(&["<pad>", "|", "A", "B", "b"]);
  assert_eq!(
    load(&mixed, &staged),
    AlignerError::Tokenization(TokenizationError::UpperWithLowercase('b'))
  );
  assert_eq!(
    load(&mixed, &as_written),
    AlignerError::Tokenization(TokenizationError::ProjectedAsWritten)
  );
  // A table the contract fits reaches the model load, which then fails on the
  // absent path.
  let plain = table(&["<pad>", "|", "A", "B"]);
  assert!(matches!(load(&plain, &staged), AlignerError::Load(_)));
}

/// **A table whose blank is spelled as a space passes the door's tokenization
/// check** (Codex R7) and reaches the model load: the blank is non-lexical, so
/// no letter or whitespace check reads it. The model path does not exist, so
/// the load is what refuses it.
#[test]
fn a_blank_spelled_as_a_space_passes_the_doors_tokenization_check() {
  let spaced_blank = table(&[" ", "|", "A", "B"]);
  let result = Aligner::from_paths_with_vocabulary(
    Lang::En,
    Path::new("/nonexistent/model.mlmodelc"),
    &spaced_blank,
    &contract(0, AcousticGeometry::WAV2VEC2),
    normalizer(),
    AlignerOptions::new(),
  );
  assert!(
    matches!(result, Err(AlignerError::Load(_))),
    "the table and the contract agree; only the absent model refuses the load"
  );
}

/// **A table whose blank is spelled `A` loads under upper case**: the door's
/// tokenization check asks for no lexical `A`, so the load reaches the model,
/// whose path does not exist. The seam built from it reserves the blank's
/// column, so a text's `a`, looked up as `A`, lands on the blank: an OOV
/// event, the policy's to decide, never a letter.
#[test]
fn a_blank_spelled_a_passes_the_doors_case_check() {
  let lettered_blank = table(&["A", "|", "B", "C"]);
  let staged = contract(0, AcousticGeometry::WAV2VEC2);
  let result = Aligner::from_paths_with_vocabulary(
    Lang::En,
    Path::new("/nonexistent/model.mlmodelc"),
    &lettered_blank,
    &staged,
    normalizer(),
    AlignerOptions::new(),
  );
  assert!(
    matches!(result, Err(AlignerError::Load(_))),
    "the table and the contract agree; only the absent model refuses the load"
  );

  let seam = build_seam(
    Lang::En,
    &lettered_blank,
    &staged,
    normalizer(),
    &AlignerOptions::new(),
  )
  .expect("builds");
  assert_eq!(
    positions(seam.detect_oov("cab").expect("detect_oov").events()),
    [(OovKind::Symbol('a'), 1, 0, Lang::En)]
  );
}

/// The staged aligner, the road `from_paths` takes.
fn staged_aligner() -> Aligner {
  Aligner::from_paths(
    Lang::En,
    &models_dir().join("base960h_aligner.mlmodelc"),
    normalizer(),
  )
  .expect("load base960h_aligner.mlmodelc (set ALIGNKIT_TEST_MODELS)")
}

/// The 11 s `jfk.wav` fixture, borrowed from the whisperkit crate by relative
/// path, failing LOUDLY if it moves.
fn jfk() -> Vec<f32> {
  let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests/whisper/fixtures/audio/jfk.wav");
  let mut reader = hound::WavReader::open(&path)
    .unwrap_or_else(|e| panic!("open the jfk.wav fixture at {path:?}: {e}"));
  assert_eq!(reader.spec().sample_rate, 16_000, "fixture must be 16 kHz");
  reader
    .samples::<i16>()
    .map(|s| f32::from(s.expect("valid sample")) / 32_768.0)
    .collect()
}

/// `text` detected by `aligner`'s seam, every event wildcarded.
fn wildcarded(aligner: &Aligner, text: &str) -> OovResolution {
  aligner
    .detect_oov(text)
    .expect("detect_oov")
    .decide(wildcard_all_policy)
}

/// **The composition keeps only real frames.** asry pads 200 real samples to
/// 400; `EncoderInput::from_prepared` reads the true pre-pad length off the
/// chunk, and the conv-geometry truncation keeps the one receptive-field frame.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn the_composition_keeps_only_real_frames() {
  let aligner = staged_aligner();
  let samples = &jfk()[80_000..80_200];
  let abort = AtomicBool::new(false);
  let prepared = aligner
    .inner
    .prepare(
      samples,
      &SpeechSpans::all_speech(),
      "test",
      wildcarded(&aligner, "test"),
      clock(),
      &abort,
    )
    .expect("prepare 200 real samples with alignable text");
  assert!(!prepared.is_trivial());
  assert_eq!(prepared.encoder_input().len(), 400, "asry pads to 400");
  let output = aligner
    .encoder
    .emissions(EncoderInput::from_prepared(&prepared))
    .expect("emissions on the prepared chunk");
  assert_eq!(crate::audio::align::encode::output_shape(&output).0, 1);
}

/// **641 samples of `ABC` have no alignment path through the composition.**
/// They truncate to one frame (`floor((641 − 400) / 320) + 1`), and one frame
/// cannot carry three distinct tokens, so `finish` returns `NoAlignmentPath` —
/// the old `ceil(641/320) = 3` threaded a plausible alignment through two
/// phantom frames.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn the_composition_of_641_samples_of_abc_has_no_alignment_path() {
  let aligner = staged_aligner();
  let samples = &jfk()[80_000..80_641];
  let text = "ABC";
  assert!(
    aligner
      .detect_oov(text)
      .expect("detect_oov")
      .events()
      .is_empty()
  );
  let abort = AtomicBool::new(false);
  let prepared = aligner
    .inner
    .prepare(
      samples,
      &SpeechSpans::all_speech(),
      text,
      wildcarded(&aligner, text),
      clock(),
      &abort,
    )
    .expect("prepare 641 samples of ABC");
  let emissions = prepared
    .encode_with(|_| {
      aligner
        .encoder
        .emissions(EncoderInput::from_prepared(&prepared))
    })
    .expect("emissions on the 641-sample chunk");
  assert_eq!(emissions.frames(), 1);
  let err = aligner
    .inner
    .finish(prepared, emissions, &abort)
    .expect_err("one frame cannot carry three distinct tokens");
  assert!(matches!(err, EmissionsError::NoAlignmentPath(_)), "{err:?}");
}

/// **An empty chunk with tokens is the named `NoAlignmentPath` through
/// `align_chunk`**, never asry's stride mismatch: the check runs before the
/// encoder, so deleting its call in `align_chunk` turns this red.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn align_chunk_names_an_empty_chunk_with_tokens() {
  let aligner = staged_aligner();
  let abort = AtomicBool::new(false);
  let err = aligner
    .align_chunk(
      &[],
      &[],
      "test",
      clock(),
      &abort,
      wildcarded(&aligner, "test"),
    )
    .expect_err("no frame can carry a token");
  assert!(matches!(err, AlignError::NoAlignmentPath(_)), "{err:?}");
}

/// **`align_chunk` is the composition, bit for bit, under a partial VAD mask**
/// — the regime where the prepared buffer differs from the raw samples, so
/// only `EncoderInput::from_prepared` at `align_chunk`'s call site reproduces
/// the composition: a mutant that encodes the raw samples instead diverges
/// here.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn align_chunk_is_the_composition_under_a_partial_vad_mask() {
  let aligner = staged_aligner();
  let samples = jfk();
  let sub_segments = [
    TimeRange::new(0, 84_000, asry::time::ANALYSIS_TIMEBASE),
    TimeRange::new(120_000, 176_000, asry::time::ANALYSIS_TIMEBASE),
  ];
  let text = "And so my fellow Americans ask not what your country can do for you, ask what you \
              can do for your country.";
  let decided = || {
    aligner
      .detect_oov(text)
      .expect("oov")
      .decide(default_oov_policy)
  };
  let abort = AtomicBool::new(false);

  let left = aligner
    .align_chunk(&samples, &sub_segments, text, clock(), &abort, decided())
    .expect("align_chunk")
    .words()
    .to_vec();

  let speech = SpeechSpans::from_time_ranges(&sub_segments).expect("speech spans");
  let prepared = aligner
    .inner
    .prepare(&samples, &speech, text, decided(), clock(), &abort)
    .expect("prepare");
  let emissions = prepared
    .encode_with(|_| {
      aligner
        .encoder
        .emissions(EncoderInput::from_prepared(&prepared))
    })
    .expect("emissions");
  let right = aligner
    .inner
    .finish(prepared, emissions, &abort)
    .expect("finish")
    .words()
    .to_vec();

  assert!(!right.is_empty(), "the composition must produce words");
  assert_eq!(left.len(), right.len());
  for (l, r) in left.iter().zip(&right) {
    assert_eq!(l.text(), r.text());
    assert_eq!(
      (l.range().start_pts(), l.range().end_pts()),
      (r.range().start_pts(), r.range().end_pts()),
      "word `{}`",
      l.text()
    );
    assert_eq!(
      l.score().to_bits(),
      r.score().to_bits(),
      "word `{}`",
      l.text()
    );
  }
}

// ---------------------------------------------------------------------
// The `tracing` feature actually emits spans.
//
// The feature was declared in Cargo.toml, advertised in `lib.rs` ("structured
// spans over load and per-chunk alignment") and implemented NOWHERE: not one
// `tracing::` call-site existed anywhere in `src/`. A user who built
// `--features tracing` with a subscriber installed got zero spans and lost the
// afternoon to their own setup.
//
// Every gate missed it, and the reason generalises: `cargo hack check
// --each-feature` only COMPILES each feature, and an unused optional dependency
// compiles perfectly clean. Only EXECUTING a test under the feature can see
// this class of bug, so these must run under `cargo hack test --each-feature` —
// which means the load half below is deliberately hermetic (a missing model
// still opens the span, because `#[instrument]` opens it before the body runs).
// ---------------------------------------------------------------------

#[cfg(feature = "tracing")]
mod tracing_spans {
  use core::cell::RefCell;
  use std::sync::{
    Once,
    atomic::{AtomicU64, Ordering},
  };

  use super::*;

  // Why a GLOBAL subscriber with thread-local capture, and not the obvious
  // `with_default(subscriber, || ...)`:
  //
  // `tracing` caches an `Interest` per callsite, PROCESS-WIDE, the first time
  // that callsite is reached. A callsite first reached while no subscriber is
  // installed caches `Interest::never()` and is then dead for the rest of the
  // process. `with_default` installs a THREAD-LOCAL subscriber and does NOT
  // rebuild that cache (`rebuild_interest_cache` recomputes from the list of
  // GLOBAL dispatchers, which is empty in that case — it cannot help). Other
  // tests in this binary call `Encoder::emissions` with no subscriber at all,
  // so on a full `--ignored` run they killed the `alignkit.encoder.emissions`
  // callsite before this test ever ran and it captured three of its four spans.
  // Found exactly that way: green alone, red in the suite.
  //
  // `set_global_default` DOES rebuild the interest cache, and `enabled()` below
  // is unconditionally true, so every callsite lands on `Interest::always()` and
  // can never be re-poisoned. Capture is then armed per-thread, which is also
  // what keeps parallel tests from seeing each other's spans.
  //
  // This is a property of scoped subscribers in a multi-test process, NOT of
  // this crate: a real user calls `init()` / `set_global_default`, which takes
  // the same path this does. There is nothing to fix in the library.

  thread_local! {
    /// `Some` while this thread is capturing; the spans it has collected.
    static CAPTURED: RefCell<Option<Capture>> = const { RefCell::new(None) };
  }

  /// One captured span: its NAME, the NAMES of its declared diagnostic fields,
  /// and the id of its (contextual) parent — everything the `tracing` contract
  /// documents, so the gate proves the fields and the nesting, not merely the
  /// count.
  #[derive(Debug)]
  struct CapturedSpan {
    id: u64,
    name: &'static str,
    fields: Vec<&'static str>,
    parent: Option<u64>,
  }

  /// The per-thread capture buffer: the spans opened so far, and the stack of
  /// currently-entered span ids that resolves each new span's CONTEXTUAL parent
  /// — exactly as `tracing-subscriber`'s registry does, since a `#[instrument]`
  /// span's parent is whichever span is entered when it is created.
  #[derive(Debug, Default)]
  struct Capture {
    spans: Vec<CapturedSpan>,
    entered: Vec<u64>,
  }

  /// A capturing [`tracing::Subscriber`]: records the NAME, FIELD names, and
  /// (contextual) PARENT of every span opened on a thread that has armed
  /// [`CAPTURED`], and ignores every other thread.
  ///
  /// Hand-rolled rather than `tracing-subscriber`: the whole surface is seven
  /// trait methods, and a test-only dependency on a second tracing crate (with
  /// its own feature matrix) to assert the nesting and fields would cost more
  /// than it explains.
  struct CaptureSpans {
    next_id: AtomicU64,
  }

  impl tracing::Subscriber for CaptureSpans {
    fn enabled(&self, _metadata: &tracing::Metadata<'_>) -> bool {
      // Every level: the load spans are INFO and the per-chunk ones DEBUG, and
      // this test is about whether they EXIST, not about filtering. Being
      // unconditional is also what pins every callsite to `Interest::always()`.
      true
    }

    fn new_span(&self, span: &tracing::span::Attributes<'_>) -> tracing::span::Id {
      // `Id::from_u64` rejects 0; `next_id` starts at 1.
      let id = self.next_id.fetch_add(1, Ordering::Relaxed);
      CAPTURED.with(|captured| {
        if let Some(capture) = captured.borrow_mut().as_mut() {
          // A `#[instrument]` span sets no explicit parent, so its parent is the
          // span currently entered on this thread (top of the stack), or none at
          // the root.
          let parent = capture.entered.last().copied();
          let fields = span
            .metadata()
            .fields()
            .iter()
            .map(|field| field.name())
            .collect();
          capture.spans.push(CapturedSpan {
            id,
            name: span.metadata().name(),
            fields,
            parent,
          });
        }
      });
      tracing::span::Id::from_u64(id)
    }

    fn record(&self, _span: &tracing::span::Id, _values: &tracing::span::Record<'_>) {}
    fn record_follows_from(&self, _span: &tracing::span::Id, _follows: &tracing::span::Id) {}
    fn event(&self, _event: &tracing::Event<'_>) {}

    fn enter(&self, span: &tracing::span::Id) {
      CAPTURED.with(|captured| {
        if let Some(capture) = captured.borrow_mut().as_mut() {
          capture.entered.push(span.into_u64());
        }
      });
    }

    fn exit(&self, span: &tracing::span::Id) {
      CAPTURED.with(|captured| {
        if let Some(capture) = captured.borrow_mut().as_mut() {
          // `#[instrument]` enters/exits are strictly nested, so the exiting span
          // is the top on the happy path; find-and-remove anyway so an unexpected
          // order cannot corrupt the stack.
          if let Some(pos) = capture
            .entered
            .iter()
            .rposition(|&id| id == span.into_u64())
          {
            capture.entered.remove(pos);
          }
        }
      });
    }
  }

  /// Runs `body` with the capturing subscriber armed on this thread and returns
  /// every span it opened, in order.
  fn spans_opened_by(body: impl FnOnce()) -> Vec<CapturedSpan> {
    static INSTALL: Once = Once::new();
    INSTALL.call_once(|| {
      tracing::subscriber::set_global_default(CaptureSpans {
        next_id: AtomicU64::new(1),
      })
      .expect("no other subscriber may claim the global default in this test binary");
    });

    CAPTURED.with(|captured| *captured.borrow_mut() = Some(Capture::default()));
    body();
    CAPTURED.with(|captured| {
      captured
        .borrow_mut()
        .take()
        .expect("capture was armed above")
        .spans
    })
  }

  fn count(spans: &[CapturedSpan], name: &str) -> usize {
    spans.iter().filter(|span| span.name == name).count()
  }

  /// The first captured span with `name` — for the field / parent assertions.
  fn first<'a>(spans: &'a [CapturedSpan], name: &str) -> &'a CapturedSpan {
    spans
      .iter()
      .find(|span| span.name == name)
      .unwrap_or_else(|| panic!("no `{name}` span was captured; got {spans:?}"))
  }

  /// The name of `span`'s parent, resolved through the captured ids.
  fn parent_name<'a>(spans: &'a [CapturedSpan], span: &CapturedSpan) -> Option<&'a str> {
    let parent = span.parent?;
    spans
      .iter()
      .find(|candidate| candidate.id == parent)
      .map(|candidate| candidate.name)
  }

  /// Asserts `span` declares every documented field in `expected`.
  fn assert_has_fields(span: &CapturedSpan, expected: &[&str]) {
    for field in expected {
      assert!(
        span.fields.contains(field),
        "`{}` span must carry the documented `{field}` field; got {:?}",
        span.name,
        span.fields
      );
    }
  }

  /// **HERMETIC, and that is the point**: this runs under `cargo hack test
  /// --each-feature`, the only gate that can see the feature do nothing.
  ///
  /// `#[instrument]` opens the span before the function body runs, so a load
  /// that FAILS still emits one — which lets the load half of the contract be
  /// proven with no CoreML model at all. The `Err` is asserted too: without it
  /// this test would keep passing if the model path silently started resolving
  /// to something real.
  #[test]
  fn load_emits_a_span_even_when_the_model_is_missing() {
    let spans = spans_opened_by(|| {
      let result = Aligner::from_paths_with(
        Lang::En,
        Path::new("/nonexistent/base960h_aligner.mlmodelc"),
        normalizer(),
        AlignerOptions::new(),
      );
      assert!(
        matches!(result, Err(AlignerError::Load(_))),
        "the point of this path is that it fails; a load that succeeded would prove nothing \
         about the span"
      );
    });

    // The span NAMES are the feature's observable contract — a subscriber
    // filters and groups on them — so they are asserted as literals here rather
    // than read back from a constant that a rename would silently carry along.
    assert!(
      count(&spans, "alignkit.aligner.load") >= 1,
      "`--features tracing` must emit a load span; got {spans:?}"
    );
    assert!(
      count(&spans, "alignkit.encoder.load") >= 1,
      "the CoreML load must be its own nested span (it is where the wall-clock hides — 308 s on \
       a cold ANE placement); got {spans:?}"
    );

    // The documented diagnostic FIELDS (a subscriber renders these) — stripping
    // any one must fail here, not slip past a name count.
    assert_has_fields(
      first(&spans, "alignkit.aligner.load"),
      &["language", "model_path", "compute"],
    );
    assert_has_fields(first(&spans, "alignkit.encoder.load"), &["path", "compute"]);

    // NESTING (aligner/mod.rs:303): the CoreML load is a CHILD of the aligner
    // load — this holds even on the failing path, since `#[instrument]` opens
    // both spans before either body runs. A load moved out from under the aligner
    // span would keep both counts but lose this parent link.
    assert_eq!(
      parent_name(&spans, first(&spans, "alignkit.encoder.load")),
      Some("alignkit.aligner.load"),
      "`alignkit.encoder.load` must nest inside `alignkit.aligner.load`; got {spans:?}"
    );
  }

  /// The per-chunk half: **one `alignkit.align_chunk` span per call**, with the
  /// CoreML predict nested inside it. Model-gated, because a span over an
  /// alignment needs an alignment — so it runs ONLY under
  /// `cargo test -p coremlit --features align,tracing -- --ignored`, the one gate that
  /// both enables `tracing` and runs `#[ignore]` tests. `cargo hack test
  /// --each-feature` enables the feature but skips ignored tests; the plain
  /// `--ignored` runs do not enable `tracing`. Without that gate in the matrix,
  /// deleting the per-call `#[instrument]` attributes on `Aligner::align_chunk`
  /// or `Encoder::emissions` would be caught by nothing at all (F4) — see the
  /// crate-root "Gates" section, which now lists it.
  #[test]
  #[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
  fn every_align_chunk_call_opens_exactly_one_span() {
    let samples = load_jfk_wav();
    let text = "And so my fellow Americans ask not what your country can do for you, ask what \
                you can do for your country.";

    let spans = spans_opened_by(|| {
      let aligner = Aligner::from_paths(
        Lang::En,
        &models_dir().join("base960h_aligner.mlmodelc"),
        normalizer(),
      )
      .expect("load base960h_aligner.mlmodelc (set ALIGNKIT_TEST_MODELS)");

      let abort = AtomicBool::new(false);

      // TWICE: "at least one span" would also pass against an `#[instrument]`
      // that somehow fired once per Aligner rather than once per chunk.
      for _ in 0..2 {
        // A resolution applies once: each chunk is detected and decided anew.
        let resolution = aligner
          .detect_oov(text)
          .expect("detect_oov")
          .decide(default_oov_policy);
        let clock = OutputClock::new(0, asry::time::ANALYSIS_TIMEBASE, 0).expect("clock");
        let result = aligner
          .align_chunk(&samples, &[], text, clock, &abort, resolution)
          .expect("align_chunk on the shipping default");
        assert!(!result.words().is_empty(), "jfk.wav must align to words");
      }
    });

    assert_eq!(
      count(&spans, "alignkit.align_chunk"),
      2,
      "one span per align_chunk call, no more and no fewer; got {spans:?}"
    );
    // EXACTLY two, tightened from `>= 2`: one predict per chunk — a second predict
    // inside a chunk would be a real regression, and zero on a chunk would be the
    // nesting bug the loop below catches.
    assert_eq!(
      count(&spans, "alignkit.encoder.emissions"),
      2,
      "exactly one CoreML predict span per chunk, two over two calls; got {spans:?}"
    );
    assert!(
      count(&spans, "alignkit.aligner.load") >= 1,
      "load must still be spanned on the success path; got {spans:?}"
    );

    // The documented diagnostic FIELDS: the per-chunk debugger aids
    // (aligner/mod.rs:427 — `sub_segments` / `text_bytes` / `samples` tell the two
    // empty-result paths apart) and the encoder's geometry/placement.
    assert_has_fields(
      first(&spans, "alignkit.align_chunk"),
      &[
        "language",
        "samples",
        "sub_segments",
        "text_bytes",
        "oov_decisions",
      ],
    );
    assert_has_fields(
      first(&spans, "alignkit.encoder.emissions"),
      &["encoder_input", "real_samples", "compute"],
    );

    // NESTING (aligner/mod.rs:424): EVERY emissions predict runs INSIDE an
    // align_chunk pass, not beside it. Moving `align_chunk` onto a helper invoked
    // after prediction would preserve both counts while un-nesting the predict —
    // this loop is what catches that.
    for span in spans
      .iter()
      .filter(|span| span.name == "alignkit.encoder.emissions")
    {
      assert_eq!(
        parent_name(&spans, span),
        Some("alignkit.align_chunk"),
        "each `alignkit.encoder.emissions` must nest inside `alignkit.align_chunk`; got {spans:?}"
      );
    }
  }

  /// `ALIGNKIT_TEST_MODELS`, or `<workspace>/Models/alignkit` — the crate's
  /// convention (`tests/common/mod.rs`), duplicated here because a `src/` unit
  /// test cannot import the `tests/` integration crate (the same duplication,
  /// for the same reason, as `encode::tests` and `registry::tests`).
  fn models_dir() -> std::path::PathBuf {
    std::env::var_os("ALIGNKIT_TEST_MODELS").map_or_else(
      || crate::tests::models_root().join("alignkit"),
      std::path::PathBuf::from,
    )
  }

  /// The 11 s `jfk.wav` fixture, borrowed from the whisperkit crate by relative
  /// path (as `encode::tests` does) and failing LOUDLY if it ever moves.
  fn load_jfk_wav() -> Vec<f32> {
    let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
      .join("tests/whisper/fixtures/audio/jfk.wav");
    let mut reader = hound::WavReader::open(&path)
      .unwrap_or_else(|e| panic!("open the jfk.wav fixture at {path:?}: {e}"));
    assert_eq!(reader.spec().sample_rate, 16_000, "fixture must be 16 kHz");
    reader
      .samples::<i16>()
      .map(|s| f32::from(s.expect("valid sample")) / 32_768.0)
      .collect()
  }
}
