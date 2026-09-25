use super::*;

use core::num::NonZeroU32;

use asry::emissions::{EmissionsFailure, EnglishNormalizer, OovKind};

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
  // words: it fixes `T`, and asry maps boundaries by the effective
  // `n_samples / (T - 1)` ratio. The seam must be HANDED that same number — not
  // because a seam-only mismatch re-times anything (at `T >= 2` it does not; the
  // ratio follows the encoder's `T`, not the hop), but because it would
  // otherwise DECLARE a stride the encoder never used, and asry does not
  // reconcile the two: `validate_stride_extent` allows `chunk_extent ± 2·hop`,
  // which on jfk.wav accepts 319, 320 AND 321 without error.
  //
  // It holds by construction: `build_seam` reads the stride off the same
  // contract the encoder truncates by. A mutant that re-spells the seam's
  // stride as the staged 320 fails the second and third cases.
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
/// 0.2 (asry#21) asks the vocabulary instead, so `é` (the normalizer keeps
/// diacritics), `&` and `4` are three `Symbol` events at their positions in the
/// normalized text.
#[test]
fn a_character_the_bundled_table_cannot_spell_is_an_oov_event() {
  let seam = build_seam(
    Lang::En,
    &Vocabulary::bundled(),
    &AcousticContract::BASE960H,
    normalizer(),
    &AlignerOptions::new(),
  )
  .expect("builds");
  let events = seam
    .detect_oov("Café AT&T b4d")
    .expect("a character the vocabulary cannot spell is an event, never an error");
  assert_eq!(
    events,
    vec![
      OovEvent::new(OovKind::Symbol('é'), 3, 0, Lang::En),
      OovEvent::new(OovKind::Symbol('&'), 7, 1, Lang::En),
      OovEvent::new(OovKind::Symbol('4'), 11, 2, Lang::En),
    ]
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

/// One decision of `decision` for `kind` at `char_index` in word `word_index`.
fn decided(
  kind: OovKind,
  char_index: usize,
  word_index: usize,
  decision: OovDecision,
) -> ResolvedOov {
  ResolvedOov::new(
    OovEvent::new(kind, char_index, word_index, Lang::En),
    decision,
  )
}

/// **A fail-closed refusal is named, and names every refused position.** The
/// refusal used to come back as an EMPTY result, the same answer as a chunk the
/// lattice cannot align and a chunk with nothing to align. It is
/// `AlignError::Refused` now, carrying the events the caller's decisions
/// resolved `FailClosed` — both of them, in the caller's order, the boundary
/// mark (whose character the normalizer removed) included — and not the ones
/// it chose to wildcard.
#[test]
fn seam_error_names_a_refusal_by_every_refused_position() {
  let decisions = [
    decided(OovKind::Symbol('é'), 3, 0, OovDecision::Wildcard),
    decided(OovKind::Symbol('&'), 7, 1, OovDecision::FailClosed),
    decided(OovKind::Symbol('4'), 11, 2, OovDecision::Wildcard),
    decided(OovKind::BoundaryPunct, 13, 2, OovDecision::FailClosed),
  ];
  let err = seam_error(
    EmissionsError::SemanticOutOfVocab(failure("OOV '&' resolved as FailClosed")),
    &decisions,
  );
  let AlignError::Refused(refusal) = err else {
    panic!("a fail-closed refusal must be AlignError::Refused, got {err:?}");
  };
  assert_eq!(
    refusal.events(),
    [decisions[1].event().clone(), decisions[3].event().clone()]
  );
  assert_eq!(
    refusal.to_string(),
    "'&' (word 1), a boundary mark (word 2)"
  );
}

/// **An unalignable chunk is the other named case.** `NoAlignmentPath` is
/// `AlignError::NoAlignmentPath` carrying asry's diagnostic — whatever the
/// decisions say, since it is the seam's error that names the case, not the
/// caller's policy (fail-closed decisions are present here on purpose).
#[test]
fn seam_error_names_a_chunk_with_no_alignment_path() {
  let decisions = [decided(OovKind::Symbol('&'), 7, 1, OovDecision::FailClosed)];
  let err = seam_error(
    EmissionsError::NoAlignmentPath(failure("no finite path")),
    &decisions,
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
  let decisions = [decided(OovKind::Symbol('4'), 1, 0, OovDecision::Wildcard)];
  assert!(matches!(
    seam_error(
      EmissionsError::SemanticOutOfVocab(failure("fail-closed OOV")),
      &decisions
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
/// (asry 0.2 reports it rather than failing on it). `prepare` needs no model —
/// the refusal is decided before the encoder would run — and what reaches the
/// caller is the named refusal of exactly that position. The same text under a
/// policy that wildcards everything prepares a chunk to align.
#[test]
fn a_fail_closed_decision_is_a_named_refusal_through_the_seam() {
  let seam = build_seam(
    Lang::En,
    &Vocabulary::bundled(),
    &AcousticContract::BASE960H,
    normalizer(),
    &AlignerOptions::new(),
  )
  .expect("builds");
  let text = "Café AT&T b4d";
  let events = seam.detect_oov(text).expect("detect_oov");
  let samples = vec![0.0f32; 16_000];
  let abort = AtomicBool::new(false);

  let decisions = asry::emissions::default_oov_decisions(&events);
  let err = seam
    .prepare(
      &samples,
      &SpeechSpans::all_speech(),
      text,
      &decisions,
      &abort,
    )
    .err()
    .map(|err| seam_error(err, &decisions))
    .expect("the default policy fails closed on `&`");
  let AlignError::Refused(refusal) = err else {
    panic!("expected the named refusal, got {err:?}");
  };
  assert_eq!(
    refusal.events(),
    [OovEvent::new(OovKind::Symbol('&'), 7, 1, Lang::En)]
  );

  let wildcards = asry::emissions::wildcard_all_decisions(&events);
  let prepared = seam
    .prepare(
      &samples,
      &SpeechSpans::all_speech(),
      text,
      &wildcards,
      &abort,
    )
    .expect("a character the policy wildcards is aligned around, not refused");
  assert!(!prepared.is_trivial());
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
    let events = bundled.detect_oov(text).expect("detect_oov");
    assert_eq!(
      own.detect_oov(text).expect("detect_oov"),
      events,
      "{text:?}"
    );
    let decisions = asry::emissions::wildcard_all_decisions(&events);
    let prepare = |seam: &EmissionsAligner| {
      seam
        .prepare(
          &samples,
          &SpeechSpans::all_speech(),
          text,
          &decisions,
          &abort,
        )
        .map(|prepared| (prepared.is_trivial(), prepared.encoder_input().to_vec()))
    };
    assert_eq!(prepare(&own), prepare(&bundled), "{text:?}");
  }
}

/// A table of another width builds its own seam under its own contract — the
/// groundwork a per-language aligner stands on. HuggingFace's 32-class
/// `wav2vec2-base-960h` table keeps `<unk>` as an entry, and its `config.json`
/// names the blank `pad_token_id: 0`, the `<pad>` entry, which its contract
/// states.
#[test]
fn a_table_of_another_width_builds_a_seam_of_that_width() {
  let table = br#"{"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6,
    "A": 7, "O": 8, "N": 9, "I": 10, "H": 11, "S": 12, "R": 13, "D": 14, "L": 15, "U": 16,
    "M": 17, "W": 18, "C": 19, "F": 20, "G": 21, "Y": 22, "P": 23, "B": 24, "V": 25, "K": 26,
    "'": 27, "X": 28, "J": 29, "Q": 30, "Z": 31}"#;
  let vocabulary = Vocabulary::from_json(table).expect("the table reads");
  let contract = contract(0, AcousticGeometry::WAV2VEC2);
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
    seam.detect_oov("b4d").expect("detect_oov"),
    [OovEvent::new(OovKind::Symbol('4'), 1, 0, Lang::En)]
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
  let guessed = EmissionsAligner::builder(Lang::En, vocabulary.tokenizer_json())
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

/// **A tokenization asry cannot honour is refused through the public door,
/// before the model loads.** The model path does not exist: the refusal is the
/// table's, the normalizer's and the contract's, decided at load before any
/// model is read — a `|`-containing space-delimited table, and the `A`/`B`/`b`
/// table under both case statements.
#[test]
fn the_door_refuses_a_tokenization_asry_cannot_honour_before_the_model_loads() {
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
    .prepare(samples, &SpeechSpans::all_speech(), "test", &[], &abort)
    .expect("prepare 200 real samples with alignable text");
  assert!(!prepared.is_trivial());
  assert_eq!(prepared.encoder_input().len(), 400, "asry pads to 400");
  let emissions = aligner
    .encoder
    .emissions(EncoderInput::from_prepared(&prepared))
    .expect("emissions on the prepared chunk");
  assert_eq!(emissions.frames(), 1);
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
  assert!(aligner.detect_oov(text).expect("detect_oov").is_empty());
  let abort = AtomicBool::new(false);
  let prepared = aligner
    .inner
    .prepare(samples, &SpeechSpans::all_speech(), text, &[], &abort)
    .expect("prepare 641 samples of ABC");
  let emissions = aligner
    .encoder
    .emissions(EncoderInput::from_prepared(&prepared))
    .expect("emissions on the 641-sample chunk");
  assert_eq!(emissions.frames(), 1);
  let clock = OutputClock::new(0, asry::time::ANALYSIS_TIMEBASE, 0).expect("clock");
  let err = aligner
    .inner
    .finish(prepared, &emissions, clock, &abort)
    .expect_err("one frame cannot carry three distinct tokens");
  assert!(matches!(err, EmissionsError::NoAlignmentPath(_)), "{err:?}");
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
  let decisions = asry::emissions::default_oov_decisions(&aligner.detect_oov(text).expect("oov"));
  let abort = AtomicBool::new(false);
  let clock = || OutputClock::new(0, asry::time::ANALYSIS_TIMEBASE, 0).expect("clock");

  let left = aligner
    .align_chunk(&samples, &sub_segments, text, clock(), &abort, &decisions)
    .expect("align_chunk")
    .words()
    .to_vec();

  let speech = SpeechSpans::from_time_ranges(&sub_segments).expect("speech spans");
  let prepared = aligner
    .inner
    .prepare(&samples, &speech, text, &decisions, &abort)
    .expect("prepare");
  let emissions = aligner
    .encoder
    .emissions(EncoderInput::from_prepared(&prepared))
    .expect("emissions");
  let right = aligner
    .inner
    .finish(prepared, &emissions, clock(), &abort)
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

      let events = aligner.detect_oov(text).expect("detect_oov");
      let decisions = asry::emissions::default_oov_decisions(&events);
      let abort = AtomicBool::new(false);

      // TWICE: "at least one span" would also pass against an `#[instrument]`
      // that somehow fired once per Aligner rather than once per chunk.
      for _ in 0..2 {
        let clock = OutputClock::new(0, asry::time::ANALYSIS_TIMEBASE, 0).expect("clock");
        let result = aligner
          .align_chunk(&samples, &[], text, clock, &abort, &decisions)
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
