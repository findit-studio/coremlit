//! End-to-end model-gated forced alignment: alignkit's [`Aligner`] drives
//! one real chunk (`jfk.wav` + its known transcript) through
//! prepare → CoreML encode → finish, proving the whole pipeline against the
//! merged asry emissions seam produces monotonic per-word timings inside the
//! audio.
//!
//! This is Gate-1 (word-timing) PLUMBING, not the parity gate: it asserts
//! alignkit's OWN output is well-formed, NOT that it agrees with asry-ort
//! (that oracle comparison is Task B5).
//!
//! # This test does not skip
//!
//! `#[ignore]` is the opt-in gate, and it is the ONLY gate. A missing model or
//! a missing fixture is a hard FAILURE, never an early `return`. It used to
//! self-skip on both, which made it a fake gate: pointing
//! `ALIGNKIT_TEST_MODELS` at an empty directory reported `test result: ok. 1
//! passed` — green, having aligned nothing. The live exposure was the fixture,
//! reached by a cross-crate relative path into whisperkit
//! (`tests/common/mod.rs`): had that file ever moved, B4's only end-to-end
//! proof would have evaporated silently while the gate stayed green. A skip
//! that looks like a pass in the test summary is worse than no test.

mod common;

use core::sync::atomic::AtomicBool;

use std::collections::BTreeMap;

use core::num::NonZeroU32;

use coremlit::audio::align::{
  ANALYSIS_TIMEBASE, AcousticContract, AcousticGeometry, AlignError, Aligner, AlignerError,
  AlignerOptions, EnglishNormalizer, Lang, OovEvent, OovKind, OutputClock, Vocabulary, Word,
  default_oov_decisions,
};

/// Builds the aligner and drives one real chunk (`jfk.wav` + its known
/// transcript) end-to-end, on the crate's shipping configuration.
///
/// `Aligner::from_paths` → `AlignerOptions::new()` → `DEFAULT_ENCODER_COMPUTE`.
/// Deliberately NOT a hardcoded compute placement: these are the crate's only
/// end-to-end proofs, so they must run the configuration that actually ships.
/// A gate pinned to a compute unit proves only that compute unit — the previous
/// default (`ComputeUnits::All`) corrupted every emission tensor while every
/// model-gated test, each pinned to `CpuOnly`, stayed green.
fn align_jfk(samples: &[f32]) -> Vec<Word> {
  let aligner = Aligner::from_paths(
    Lang::En,
    &common::model_path(),
    Box::new(EnglishNormalizer::new()),
  )
  .expect(
    "build the En aligner from the CoreML model + bundled tokenizer (set ALIGNKIT_TEST_MODELS \
     to the model directory)",
  );

  let text = common::JFK_TRANSCRIPT;
  let events = aligner.detect_oov(text).expect("detect_oov");
  let decisions = default_oov_decisions(&events);

  // No VAD (whole chunk is speech). OutputClock anchored at stream sample 0
  // in the 1/16000 analysis timebase, so per-word PTS ARE 16 kHz sample
  // indices — directly comparable to the audio length.
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock construction");
  let abort = AtomicBool::new(false);

  aligner
    .align_chunk(samples, &[], text, clock, &abort, &decisions)
    .expect("align_chunk succeeds end-to-end")
    .words()
    .to_vec()
}

#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn align_chunk_produces_monotonic_word_timings() {
  let samples = common::load_wav_mono_f32(&common::jfk_wav_path());
  assert!(!samples.is_empty(), "fixture decoded to no samples");

  let words = &align_jfk(&samples);
  assert!(
    !words.is_empty(),
    "a real transcript over matching audio must produce words"
  );

  let bound = samples.len() as i64;
  let mut prev_start = 0_i64;
  for word in words {
    let range = word.range();
    let (start, end) = (range.start_pts(), range.end_pts());
    assert!(
      start <= end,
      "word `{}`: start {start} exceeds end {end}",
      word.text()
    );
    assert!(
      start >= 0 && end <= bound,
      "word `{}`: range [{start}, {end}] escapes the audio [0, {bound}]",
      word.text()
    );
    assert!(
      start >= prev_start,
      "word `{}`: start {start} precedes the previous word's start {prev_start} (not monotonic)",
      word.text()
    );
    let score = word.score();
    assert!(
      (0.0..=1.0).contains(&score),
      "word `{}`: score {score} outside [0, 1]",
      word.text()
    );
    prev_start = start;
  }
}

/// **Gate 3 — determinism** (design spec §7): two runs, bit-identical.
///
/// Not a tolerance and not a statistic: `assert_eq!` on the PTS integers and on
/// the score's raw bits (`f32::to_bits`, so a `NaN` or a `-0.0`/`+0.0` flip
/// cannot slip through the `==` that `f32: PartialEq` would give). Every stage
/// downstream of the encoder is deterministic dynamic programming, so this is
/// really a statement about CoreML: the same weights on the same input on the
/// same placement must return the same tensor, twice.
///
/// It matters because the alternative was live. When this model was scheduled
/// on the ANE, 16.7% of its emission cells saturated to a `-45440` sentinel —
/// and that corruption was **bit-identical run to run**. Had it instead been
/// *non-deterministic*, every other gate in this crate would have flickered
/// rather than failed, which is far harder to diagnose. This test is what says
/// which of the two we are in.
///
/// Deliberately reloads the model on each run (via [`align_jfk`]) rather than
/// reusing one `Aligner`: a load-time nondeterminism — a compute-placement
/// decision CoreML makes differently on a second load, say — is exactly the
/// class of defect a same-session double-`predict` would hide.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn align_chunk_is_bit_identical_across_runs() {
  let samples = common::load_wav_mono_f32(&common::jfk_wav_path());

  let first = align_jfk(&samples);
  let second = align_jfk(&samples);

  assert!(
    !first.is_empty(),
    "a real transcript over matching audio must produce words"
  );
  assert_eq!(
    first.len(),
    second.len(),
    "two runs over identical input produced different word counts"
  );

  for (a, b) in first.iter().zip(&second) {
    assert_eq!(a.text(), b.text(), "word text differs between runs");
    assert_eq!(
      (a.range().start_pts(), a.range().end_pts()),
      (b.range().start_pts(), b.range().end_pts()),
      "word `{}`: timing differs between two runs over identical input",
      a.text()
    );
    assert_eq!(
      a.score().to_bits(),
      b.score().to_bits(),
      "word `{}`: score differs between two runs over identical input ({} vs {})",
      a.text(),
      a.score(),
      b.score()
    );
  }
}

/// **The codex-fence regression, on the canonical `Aligner` path — and a
/// genuinely unalignable chunk is a NAMED case.** 641 real samples carrying three
/// distinct tokens (`ABC`) truncate to one emission frame, and one frame cannot
/// carry three tokens — so the seam returns `NoAlignmentPath`, which
/// [`Aligner::align_chunk`] names: [`AlignError::NoAlignmentPath`], never an
/// empty result a caller could not tell from a policy refusal or from a chunk
/// with nothing to align (see `align_chunk`'s doc and `seam_error`).
///
/// This is the canonical-path half of `tests/prepared_composition.rs`'s
/// `public_prepared_composition_641_abc_has_no_alignment_path` (which pins the raw
/// `NoAlignmentPath` error at the seam). Both are mutation proofs: revert
/// `truncated_frame_count` to `ceil(641/320) = 3` and the trellis threads `ABC`
/// across three phantom, padding-derived frames, so `align_chunk` returns words
/// — failing this assertion. asry's `chunk_extent ± 2·hop` stride check
/// (`3×320 = 960` inside `641 ± 640`) is too loose to catch the phantom frames;
/// this end-to-end test is the guard.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn align_chunk_641_abc_is_a_named_no_alignment_path() {
  let aligner = Aligner::from_paths(
    Lang::En,
    &common::model_path(),
    Box::new(EnglishNormalizer::new()),
  )
  .expect("build the En aligner (set ALIGNKIT_TEST_MODELS to the model directory)");

  let jfk = common::load_wav_mono_f32(&common::jfk_wav_path());
  let samples = &jfk[80_000..80_641];
  assert_eq!(
    samples.len(),
    641,
    "the fence case is exactly 641 real samples"
  );

  let text = "ABC";
  let events = aligner.detect_oov(text).expect("detect_oov");
  assert!(
    events.is_empty(),
    "A, B, C must be in-vocab, or the OOV path — not the frame count — would drive the result"
  );
  let decisions = default_oov_decisions(&events);
  // No VAD; clock anchored at stream sample 0 in the analysis timebase.
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock construction");
  let abort = AtomicBool::new(false);

  match aligner.align_chunk(samples, &[], text, clock, &abort, &decisions) {
    Err(AlignError::NoAlignmentPath(_)) => {}
    Ok(result) => panic!(
      "one frame cannot carry three distinct tokens, yet align_chunk returned words {:?}",
      result.words().iter().map(Word::text).collect::<Vec<_>>()
    ),
    Err(err) => panic!("expected the named AlignError::NoAlignmentPath, got {err:?}"),
  }
}

/// **A fail-closed refusal is NAMED, never an empty result.** The default policy
/// refuses the `&` of `AT&T` (a pronounced symbol) and wildcards the rest; the
/// refusal comes back from [`Aligner::align_chunk`] as
/// [`AlignError::Refused`] carrying exactly that position — before the encoder
/// runs, so the audio is incidental. It used to come back as an empty word list,
/// the same answer as the no-path chunk above.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn a_fail_closed_refusal_is_named_through_align_chunk() {
  let aligner = Aligner::from_paths(
    Lang::En,
    &common::model_path(),
    Box::new(EnglishNormalizer::new()),
  )
  .expect("build the En aligner (set ALIGNKIT_TEST_MODELS to the model directory)");
  let samples = common::load_wav_mono_f32(&common::jfk_wav_path());
  let text = "ask not what your country can do for you, AT&T";

  let events = aligner.detect_oov(text).expect("detect_oov");
  let decisions = default_oov_decisions(&events);
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock construction");
  let abort = AtomicBool::new(false);
  let err = aligner
    .align_chunk(&samples, &[], text, clock, &abort, &decisions)
    .expect_err("the default policy fails closed on `&`");
  let AlignError::Refused(refusal) = err else {
    panic!("the refusal must be named, got {err:?}");
  };
  assert_eq!(
    refusal
      .events()
      .iter()
      .map(|event| event.kind().clone())
      .collect::<Vec<_>>(),
    [OovKind::Symbol('&')],
    "the refusal names the one position the policy refused"
  );
}

/// **A character the vocabulary cannot spell arrives as an OOV event through
/// the aligner, and is aligned around.** `jfk.wav` with its transcript spelled
/// `Américans`: the bundled 29-class table has no `é`, and asry 0.1 failed the
/// whole chunk on it inside `detect_oov` (`encode('é') failed:
/// MissingUnkToken`). asry 0.2 reports it — one `Symbol('é')` event, which the
/// default policy wildcards — and the chunk aligns to every word, `américans`
/// among them.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn a_character_the_vocabulary_cannot_spell_is_an_event_through_the_aligner() {
  let aligner = Aligner::from_paths(
    Lang::En,
    &common::model_path(),
    Box::new(EnglishNormalizer::new()),
  )
  .expect("build the En aligner (set ALIGNKIT_TEST_MODELS to the model directory)");
  let samples = common::load_wav_mono_f32(&common::jfk_wav_path());
  let text = common::JFK_TRANSCRIPT.replacen("Americans", "Américans", 1);

  let events = aligner
    .detect_oov(&text)
    .expect("a character the vocabulary cannot spell is an event, never an error");
  let symbols: Vec<&OovEvent> = events
    .iter()
    .filter(|event| matches!(event.kind(), OovKind::Symbol(_)))
    .collect();
  assert_eq!(symbols.len(), 1, "one unspellable character: {events:?}");
  assert_eq!(symbols[0].kind(), &OovKind::Symbol('é'));
  assert_eq!(symbols[0].word_index(), 4, "`Américans` is the fifth word");

  let decisions = default_oov_decisions(&events);
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock construction");
  let abort = AtomicBool::new(false);
  let words = aligner
    .align_chunk(&samples, &[], &text, clock, &abort, &decisions)
    .expect("the wildcarded character is aligned around")
    .words()
    .to_vec();
  assert_eq!(
    words.len(),
    align_jfk(&samples).len(),
    "every word of the transcript aligns, the one holding `é` included"
  );
  assert!(
    words
      .iter()
      .any(|word| word.text().eq_ignore_ascii_case("américans")),
    "{:?}",
    words.iter().map(Word::text).collect::<Vec<_>>()
  );
}

/// Aligns `jfk.wav` through [`Aligner::from_paths_with_vocabulary`] with
/// `vocabulary` and `contract`, on the shipping options.
fn align_jfk_with(
  samples: &[f32],
  vocabulary: &Vocabulary,
  contract: &AcousticContract,
) -> Vec<Word> {
  let aligner = Aligner::from_paths_with_vocabulary(
    Lang::En,
    &common::model_path(),
    vocabulary,
    contract,
    Box::new(EnglishNormalizer::new()),
    AlignerOptions::new(),
  )
  .expect("the model loads with its own vocabulary and contract");
  let text = common::JFK_TRANSCRIPT;
  let events = aligner.detect_oov(text).expect("detect_oov");
  let decisions = default_oov_decisions(&events);
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock construction");
  let abort = AtomicBool::new(false);
  aligner
    .align_chunk(samples, &[], text, clock, &abort, &decisions)
    .expect("align_chunk through the model's own vocabulary")
    .words()
    .to_vec()
}

/// **The staged model aligns identically through its own vocabulary and
/// through the bundled table.** `base960h_dict.json` is the table the model
/// ships beside it; read through [`Vocabulary::from_file`] and loaded with
/// [`Aligner::from_paths_with_vocabulary`] and the staged contract, it aligns
/// `jfk.wav` to exactly the words [`Aligner::from_paths`] does — same text, same
/// ticks, same score bits. So it does under a contract of its own stating the
/// same blank and geometry, which carries no sentinel band: on the default
/// placement the emissions are clean, and the band only guards. The road a
/// per-language aligner is built through is the road English already takes.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn the_staged_model_aligns_identically_through_its_own_vocabulary() {
  let samples = common::load_wav_mono_f32(&common::jfk_wav_path());
  let bundled = align_jfk(&samples);

  let vocabulary = Vocabulary::from_file(common::dict_path())
    .expect("read base960h_dict.json (set ALIGNKIT_TEST_MODELS to the model directory)");
  assert_eq!(vocabulary.size().get(), 29);
  let generic = AcousticContract::new(0, AcousticGeometry::WAV2VEC2);
  for contract in [AcousticContract::BASE960H, generic] {
    let own = align_jfk_with(&samples, &vocabulary, &contract);
    assert!(!own.is_empty(), "jfk.wav aligns to words");
    assert_eq!(own.len(), bundled.len(), "the same number of words");
    for (a, b) in own.iter().zip(&bundled) {
      assert_eq!(a.text(), b.text());
      assert_eq!(
        (a.range().start_pts(), a.range().end_pts()),
        (b.range().start_pts(), b.range().end_pts()),
        "word `{}` under {contract:?}: the two roads time it differently",
        a.text()
      );
      assert_eq!(
        a.score().to_bits(),
        b.score().to_bits(),
        "word `{}` under {contract:?}: the two roads score it differently",
        a.text()
      );
    }
  }
}

/// **A vocabulary whose width disagrees with the model is refused by name at
/// load.** Two synthetic tables built from the staged one: with an `É` entry
/// added (30 entries) and with `Z` removed (28). The model's CTC head scores 29
/// classes, so each is [`AlignerError::VocabularyMismatch`] naming both widths —
/// at construction, before any chunk could be aligned against the wrong columns.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn a_vocabulary_of_another_width_is_refused_by_name_at_load() {
  let staged = std::fs::read(common::dict_path())
    .expect("read base960h_dict.json (set ALIGNKIT_TEST_MODELS to the model directory)");
  let table: BTreeMap<String, u32> =
    serde_json::from_slice(&staged).expect("the staged table is `{token: id}` JSON");

  let mut wider = table.clone();
  wider.insert("É".to_owned(), 29);
  let mut narrower = table;
  assert_eq!(narrower.remove("Z"), Some(28), "`Z` is the last id");

  for (synthetic, entries) in [(wider, 30), (narrower, 28)] {
    let json = serde_json::to_vec(&synthetic).expect("a table serializes");
    let vocabulary = Vocabulary::from_json(&json).expect("the synthetic table is well formed");
    assert_eq!(vocabulary.size().get(), entries);
    let result = Aligner::from_paths_with_vocabulary(
      Lang::En,
      &common::model_path(),
      &vocabulary,
      &AcousticContract::BASE960H,
      Box::new(EnglishNormalizer::new()),
      AlignerOptions::new(),
    );
    let Err(AlignerError::VocabularyMismatch(mismatch)) = result else {
      panic!(
        "a {entries}-entry table on the 29-class model must be refused by name, got {:?}",
        result.err()
      );
    };
    assert_eq!((mismatch.vocabulary(), mismatch.model()), (entries, 29));
  }
}

/// Loads the staged model with `vocabulary` and `contract`, keeping only the
/// refusal.
fn load_with(vocabulary: &Vocabulary, contract: &AcousticContract) -> AlignerError {
  match Aligner::from_paths_with_vocabulary(
    Lang::En,
    &common::model_path(),
    vocabulary,
    contract,
    Box::new(EnglishNormalizer::new()),
    AlignerOptions::new(),
  ) {
    Ok(_) => panic!("{contract:?} must be refused at load"),
    Err(err) => err,
  }
}

/// **A geometry that disagrees with the model's declared frame count is refused
/// by name at load.** The staged model declares a 960,000-sample window and
/// 2999 frames. A contract stating a 160-sample stride makes 5998 frames of
/// that window, and one stating a 321-sample stride 2990: neither is this
/// model's front end, and each is [`AlignerError::FrameCountMismatch`], naming
/// the window and both counts, before any chunk is truncated to the wrong
/// number of frames.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn a_geometry_that_disagrees_with_the_declared_frame_count_is_refused_at_load() {
  let vocabulary = Vocabulary::from_file(common::dict_path())
    .expect("read base960h_dict.json (set ALIGNKIT_TEST_MODELS to the model directory)");
  for (stride, derived) in [(160u32, 5_998usize), (321, 2_990)] {
    let geometry = AcousticGeometry::new(
      16_000,
      NonZeroU32::new(400).expect("nonzero"),
      NonZeroU32::new(stride).expect("nonzero"),
    )
    .expect("a geometry");
    let AlignerError::FrameCountMismatch(mismatch) =
      load_with(&vocabulary, &AcousticContract::new(0, geometry))
    else {
      panic!("a {stride}-sample stride must be a FrameCountMismatch");
    };
    assert_eq!(
      (mismatch.window(), mismatch.declared(), mismatch.derived()),
      (960_000, 2_999, derived)
    );
  }
}

/// **An explicit blank that is not in the table is refused by name at load.**
/// The staged table's ids run `0..29`; a contract naming id 29 the blank names
/// no column of it, and the load is [`AlignerError::BlankOutOfVocabulary`],
/// naming the blank and the table's size.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn an_explicit_blank_outside_the_table_is_refused_at_load() {
  let vocabulary = Vocabulary::from_file(common::dict_path())
    .expect("read base960h_dict.json (set ALIGNKIT_TEST_MODELS to the model directory)");
  let AlignerError::BlankOutOfVocabulary(refused) = load_with(
    &vocabulary,
    &AcousticContract::new(29, AcousticGeometry::WAV2VEC2),
  ) else {
    panic!("id 29 is no id of the 29-entry table");
  };
  assert_eq!((refused.blank(), refused.entries()), (29, 29));
}
