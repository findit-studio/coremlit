//! Simulated-stream LocalAgreement-2 on jfk.wav / tiny (ports the
//! whisperkit-cli `transcribeStreamSimulated` loop, TranscribeCLI.swift:322-424).
//!
//! # Two portable properties, and one host-scoped measurement
//!
//! The PORTABLE properties are asserted on every host, because each compares
//! this run against ITSELF on the same machine — host fp16 drift moves both
//! sides of the comparison together, so neither can red for hardware reasons:
//!
//! - **Truncation, never rewriting.** [`finalize`](
//!   coremlit::audio::whisper::stream::agreement::LocalAgreementTranscriber::finalize)'s
//!   confirmed text is a word-for-word prefix of the SAME kit's batch transcript
//!   of the whole clip. Confirming fewer words than the clip contains is
//!   LocalAgreement-2 working as designed — it holds back everything two
//!   consecutive hypotheses have not yet agreed on. Confirming a word the batch
//!   decode contradicts is not.
//! - **Monotone confirmation.** Across the pushes, `confirmed_words_slice()`
//!   only grows, `last_agreed_seconds()` never decreases, and no already-
//!   confirmed word is revised.
//!
//! Both are non-vacuous by construction: the prefix check refuses a confirmed
//! text too short to have completed one agreement round (every string starts
//! with the empty string, so an unconstrained prefix check on an empty
//! confirmation asserts nothing), and the monotonicity check refuses a run in
//! which nothing was ever confirmed. The falsifiers for each are the hermetic
//! tests at the bottom of this file; they are NOT `#[ignore]`d, so the
//! predicates are gated on every host, model or no model.
//!
//! The MEASURED observation is whether the confirmed stream ever reaches the
//! clip's canonical phrase, [`CANONICAL_PHRASE`]. That is a description of one
//! machine, not a property of the port, so it rides `tests/support/measured_band.rs`'s
//! three-way host gate: asserted on [`CHARACTERIZED_ON`], computed and PRINTED
//! everywhere else.
//!
//! # Why the phrase describes a host rather than the port
//!
//! An earlier revision of this file asserted the phrase unconditionally and
//! justified it by claiming the streaming decode "drifts with exactly the same
//! fp16 argmax flips" as `whisper_parity_jfk`. **That claim was false**, and PR
//! #89's CI runner disproved it: `jfk_tiny_golden`'s tokens are byte-identical
//! on the runner and on the development machine — the BATCH decode does not
//! diverge at all — while the confirmed stream on the runner stops at
//! `"and so my fellow americans"`.
//!
//! The two paths are not the same decode with different numbers. Every stride
//! after the first re-enters the decoder with `prefix_tokens` and
//! `clip_timestamps` that the batch path never sets
//! ([`LocalAgreement::decoding_options_for_next`](
//! coremlit::audio::whisper::stream::agreement::LocalAgreement::decoding_options_for_next)),
//! and that changes which code runs, not just what it computes on. Once a
//! stride decodes `"And so my fellow Americans!"` — an exclamation mark the
//! batch decode never produces; `jfk_tiny_golden.json`'s own text has none —
//! the held-back words prefill the next window with a sentence-final `"!"`, and
//! `TimestampRulesFilter`'s timestamp-mass rule
//! (`src/audio/whisper/decode/filter/mod.rs:233`, `timestamp_mass_exceeds_text`
//! at `:306`) masks ALL text logits and forces a closing timestamp. The window
//! then re-emits only its own prefix, agrees with the previous hypothesis on
//! exactly `agreement_count_needed` words, confirms nothing, and re-establishes
//! the prefix that caused the stall. The `[stride ...]` lines this test prints
//! show it directly: a run of strides pinned at a 1.4 s watermark with three
//! confirmed words, then one stride that escapes.
//!
//! That is a BOOLEAN control-flow gate riding a sub-nat logit margin, not a
//! cascade of argmax flips, and it is sensitive to PLACEMENT on a single
//! machine. The diagnosis that established the mechanism reports the mass
//! rule's own margin at the 7 s buffer as `+0.230` (ANE, fires), `+0.280`
//! (`CpuAndGpu`, fires) and `-0.049` (`CpuOnly`, does NOT fire), with under one
//! nat of margin at the last stride that can still escape — inside this repo's
//! documented ~1.0 worst cross-placement logit delta (`parity_jfk.rs`, "Numeric
//! drift"). Those logit numbers are that diagnosis's, not this file's; what THIS
//! file measured is their consequence, and it agrees: on the host recorded in
//! [`CHARACTERIZED_ON`], `CpuOnly` escapes the stall a whole stride earlier than
//! either accelerated placement (see the table there). A gate that a placement
//! change moves on ONE machine is not a portable contract on every machine.
//!
//! # Not the golden's host
//!
//! This gate deliberately no longer borrows `jfk_tiny_golden.json`'s
//! `generationHost`. It owns no golden, and `common::golden_host_note` applies
//! GOLDEN semantics: a foreign host PANICS before any measurement. That is right
//! for a committed-oracle comparison, where the comparison is the whole test,
//! and wrong here — after the goldens were stamped with the CI runner's host it
//! hard-red this gate on every development machine without ever reaching a
//! CoreML number. The phrase's provenance is this machine's own measurement,
//! recorded in [`CHARACTERIZED_ON`].

mod common;

use std::panic::{AssertUnwindSafe, catch_unwind};

use coremlit::audio::whisper::{
  options::{DecodingOptions, Options},
  result::WordTiming,
  stream::agreement::{LocalAgreement, STRIDE_SAMPLES},
  text::normalized,
  transcribe::WhisperKit,
};

/// The clip's canonical phrase, as `whisperkit-cli` transcribes jfk.wav on
/// tiny. MEASURED, not portable: whether the confirmed stream reaches it is
/// decided by the timestamp-mass gate described in this module's docs, which a
/// same-host placement change already flips. Never weaken, widen or delete it —
/// it moves behind a host gate, it does not get easier.
const CANONICAL_PHRASE: &str = "ask not what your country can do for you";

/// The host class [`CANONICAL_PHRASE`] was measured on.
///
/// RECORDED, unlike the siglip bands — because unlike them, this one could be
/// measured rather than guessed. Sweeping the encoder and decoder together with
/// mel left on its shipping `CpuAndGpu`, every placement this machine can drive
/// confirms the WHOLE 22-word transcript, phrase included:
///
/// | encoder/decoder | phrase | confirmed | escapes the stall between strides |
/// | --- | --- | --- | --- |
/// | `CpuAndNeuralEngine` (shipping default) | present | 22/22 words | 9 -> 10 (watermark 1.44 s -> 8.68 s) |
/// | `CpuAndGpu` | present | 22/22 words | 9 -> 10 (watermark 1.44 s -> 8.68 s) |
/// | `CpuOnly` | present | 22/22 words | 8 -> 9 (watermark 1.44 s -> 6.92 s) |
///
/// So the phrase is not a knife-edge here: it survives a placement change, and
/// the only thing placement moves is which stride escapes the timestamp-mass
/// stall. On the CI runner it is not merely close — the stream stalls at
/// `"and so my fellow americans"` and never escapes at all. Arming it there
/// would assert something known to be false, which is exactly what the
/// [`common::BandVerdict::Foreign`] path exists to prevent.
///
/// To arm a different machine: run the command the band-gate banner prints, read
/// the `[band]` line, and point this at THAT host class. Never widen the phrase
/// to span both.
const CHARACTERIZED_ON: Option<common::CharacterizedHost> = Some(common::CharacterizedHost {
  os_build: "25F71",
  os_product_version: "26.5",
  chip: "Apple M1 Max",
  arch: "arm64",
});

/// The exact command that re-measures the phrase on THIS machine, quoted into
/// every band-gate banner so a log names its own fix.
fn recharacterize_command() -> String {
  "cargo test -p coremlit --features whisper --test whisper_streaming -- --ignored --nocapture\n                \
   then read the printed `[band]` line: if the phrase was present, set\n                \
   CHARACTERIZED_ON in crates/coremlit/tests/whisper/streaming.rs to the `this host`\n                \
   line above; if it was ABSENT, leave the recorded host alone — a host that does not\n                \
   produce the phrase must not arm it."
    .to_string()
}

// ---------------------------------------------------------------------
// The portable properties, as pure predicates
// ---------------------------------------------------------------------

/// One completed push's view of the confirmation state.
#[derive(Debug, Clone)]
struct Confirmation {
  /// 1-based stride index; also the buffered seconds, at a 1 s stride.
  stride: usize,
  /// [`LocalAgreement::last_agreed_seconds`] after this push.
  last_agreed_seconds: f32,
  /// [`LocalAgreement::confirmed_words_slice`], normalized — settled words.
  confirmed: Vec<String>,
  /// [`LocalAgreement::last_agreed_words_slice`], normalized — agreed but held
  /// back, still revisable by a later hypothesis.
  held_back: Vec<String>,
}

impl Confirmation {
  fn observe(stride: usize, agreement: &LocalAgreement) -> Self {
    Self {
      stride,
      last_agreed_seconds: agreement.last_agreed_seconds(),
      confirmed: normalized_words(agreement.confirmed_words_slice()),
      held_back: normalized_words(agreement.last_agreed_words_slice()),
    }
  }
}

/// Word timings as the normalized word strings the agreement compares.
fn normalized_words(words: &[WordTiming]) -> Vec<String> {
  words.iter().map(|word| normalized(word.word())).collect()
}

/// PORTABLE: `confirmed` is a word-for-word prefix of `batch`, and long enough
/// to be evidence.
///
/// `min_words` is the run's own [`LocalAgreement::agreement_count_needed`]: one
/// completed agreement round confirms at least that many words into the
/// finalized text (a round whose common prefix is exactly that long holds all of
/// them back, and `finalize` folds the holdback in), so a shorter confirmation
/// means no two consecutive hypotheses ever agreed at all. Without this floor
/// the prefix check would pass vacuously on an empty confirmation, since every
/// string starts with the empty one — a green light that asserts nothing is
/// worse than the host-specific assertion it replaced.
///
/// # Errors
/// A confirmation shorter than `min_words`, or one that rewrites rather than
/// truncates, with the first diverging word named.
fn check_confirmed_is_prefix_of_batch(
  confirmed: &str,
  batch: &str,
  min_words: usize,
) -> Result<(), String> {
  let confirmed_words: Vec<&str> = confirmed.split_whitespace().collect();
  let batch_words: Vec<&str> = batch.split_whitespace().collect();

  if confirmed_words.len() < min_words {
    return Err(format!(
      "the stream confirmed {} word(s), fewer than the {min_words} that one completed \
       agreement round puts into the finalized text.\n  \
       confirmed : {confirmed:?}\n  \
       batch     : {batch:?}\n\
       LocalAgreement-2 is ALLOWED to stop early — that is what holding words back \
       means — but reaching here says no two consecutive hypotheses agreed on \
       {min_words} words anywhere in the clip, and the prefix check below would then \
       pass vacuously (every string starts with the empty string). This is a \
       confirmation stall, not a tolerance to relax.",
      confirmed_words.len()
    ));
  }

  if batch_words.starts_with(&confirmed_words) {
    return Ok(());
  }

  let mut report = format!(
    "the confirmed stream REWRITES the batch transcript rather than truncating it.\n  \
     confirmed : {} word(s) {confirmed:?}\n  \
     batch     : {} word(s) {batch:?}\n",
    confirmed_words.len(),
    batch_words.len(),
  );
  match confirmed_words
    .iter()
    .zip(&batch_words)
    .position(|(ours, theirs)| ours != theirs)
  {
    Some(index) => report.push_str(&format!(
      "  first divergence at word {index}: confirmed {:?} vs batch {:?}\n",
      confirmed_words[index], batch_words[index],
    )),
    // Every shared position matched, so the confirmation ran PAST the batch
    // transcript — the signature of a word confirmed twice. Two mechanisms
    // produce it. `LocalAgreement::finalize` folding in BOTH the held-back words
    // and the last pair's differing suffix is now guarded (`holdback_superseded`)
    // and has hermetic falsifiers in `stream/agreement/tests.rs`. The
    // re-admission strip in `LocalAgreement::watermark_filtered` is NOT fixed —
    // it only catches a reproduction at the front of the offered list, and the
    // sequences that slide past it are the `_today` characterization tests in
    // that same file (issue #94). Check the second before the decode, and the
    // first before both.
    None => report.push_str(&format!(
      "  every shared position matched, but the confirmation is {} word(s) LONGER \
       than the batch transcript — i.e. it emitted words the batch decode never \
       produced. Suspect a word confirmed TWICE before suspecting the decode: \
       `LocalAgreement::watermark_filtered`'s re-admission strip has KNOWN OPEN \
       gaps (issue #94, the `_today` characterization tests in \
       `stream/agreement/tests.rs`), \
       and `LocalAgreement::finalize` folds in both the held-back words and the \
       last pair's differing suffix behind the `holdback_superseded` guard.\n",
      confirmed_words.len().saturating_sub(batch_words.len()),
    )),
  }
  report.push_str(
    "\nTruncating is fine and expected; contradicting a word already handed to the \
     caller is not.\nThis compares the run against ITSELF on this machine — same kit, \
     same clip, same compute\npath — so host fp16 drift moves both sides together and \
     cannot cause it. Do not add a\ntolerance: there is no number here to relax.",
  );
  Err(report)
}

/// PORTABLE: confirmation only ever moves forward across the pushes.
///
/// Checks three ways of going backwards — a receding
/// [`LocalAgreement::last_agreed_seconds`] watermark, a shrinking confirmed
/// list, and a rewritten confirmed word — and then refuses a run that confirmed
/// nothing at all, which would satisfy all three trivially.
///
/// # Errors
/// The first backwards step, naming both snapshots; or a run with no pushes or
/// no confirmation to be monotone about.
fn check_monotone_confirmation(steps: &[Confirmation]) -> Result<(), String> {
  let Some(last) = steps.last() else {
    return Err(
      "no pushes were observed, so monotonicity was asserted over an empty sequence \
       and could not have failed. The clip must produce at least one stride."
        .to_string(),
    );
  };

  for pair in steps.windows(2) {
    let (before, after) = (&pair[0], &pair[1]);
    if after.last_agreed_seconds < before.last_agreed_seconds {
      return Err(format!(
        "the agreement watermark RECEDED between stride {} and stride {}: {} s -> {} s.\n\
         Words before the watermark are settled and are never revisited, so moving it \
         back re-opens\ntext the caller was already told was final.",
        before.stride, after.stride, before.last_agreed_seconds, after.last_agreed_seconds,
      ));
    }
    if after.confirmed.len() < before.confirmed.len() {
      return Err(format!(
        "the confirmed word list SHRANK between stride {} and stride {}: {} -> {} words.\n  \
         stride {} : {:?}\n  stride {} : {:?}\n\
         Confirmed words are handed to the caller as final; retracting one is a defect, \
         not drift.",
        before.stride,
        after.stride,
        before.confirmed.len(),
        after.confirmed.len(),
        before.stride,
        before.confirmed,
        after.stride,
        after.confirmed,
      ));
    }
    if !after.confirmed.starts_with(&before.confirmed) {
      let index = before
        .confirmed
        .iter()
        .zip(&after.confirmed)
        .position(|(was, now)| was != now)
        .unwrap_or(before.confirmed.len());
      return Err(format!(
        "a CONFIRMED word was rewritten between stride {} and stride {}, at index {index}: \
         {:?} -> {:?}.\n  stride {} : {:?}\n  stride {} : {:?}\n\
         Confirmation is append-only; a later hypothesis may extend it and may never \
         revise it.",
        before.stride,
        after.stride,
        before.confirmed.get(index),
        after.confirmed.get(index),
        before.stride,
        before.confirmed,
        after.stride,
        after.confirmed,
      ));
    }
  }

  if last.confirmed.is_empty() && last.held_back.is_empty() {
    return Err(format!(
      "the stream confirmed and held back NOTHING across all {} stride(s), so every \
       monotonicity check above passed on empty sequences and asserted nothing.\n\
       A run with no confirmation is a stall to diagnose, not a monotone run.",
      steps.len(),
    ));
  }
  Ok(())
}

// ---------------------------------------------------------------------
// The model-gated measurement
// ---------------------------------------------------------------------

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn jfk_simulated_stream_confirms_the_transcript() {
  // Opened BEFORE any CoreML number, so the log leads with whether the phrase is
  // asserted here, on which host it was measured, and how to re-measure.
  let gate = common::BandGate::open(
    "whisper streaming confirmed phrase",
    CHARACTERIZED_ON,
    recharacterize_command(),
  );

  // `Options::new` takes both folders directly (two-arg constructor, not a
  // zero-arg `new()` plus `with_model_folder`/`with_tokenizer_folder`
  // builders) — same brief-vs-shipped-API fix as tests/pipeline.rs's
  // `tiny_options`/tests/parity_jfk.rs.
  let kit = WhisperKit::new(&Options::new(common::tiny_dir(), common::tokenizer_dir())).unwrap();
  let audio = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/jfk.wav"));

  // The reference for the prefix property: the SAME kit decoding the SAME clip
  // in one batch, under the same word-timestamped options the streamer forces on
  // its own copy. Holding `word_timestamps` equal is what leaves the per-stride
  // `prefix_tokens`/`clip_timestamps` retargeting as the only difference between
  // the two — which is exactly the thing under test.
  let batch_options = DecodingOptions::new().with_word_timestamps();
  let batch = normalized(kit.transcribe(&audio, &batch_options).unwrap().text());
  println!("[batch] {batch:?}");

  let mut streamer = kit.local_agreement_transcriber(DecodingOptions::new());
  // 1 s pushes — 11 strides, each re-transcribing the grown prefix.
  let mut steps = Vec::new();
  for (index, chunk) in audio.chunks(STRIDE_SAMPLES).enumerate() {
    let outcomes = streamer.push_samples(chunk).unwrap();
    let step = Confirmation::observe(index + 1, streamer.agreement());
    println!(
      "[stride {:>2}] {:<18} watermark {:>6.2} s  confirmed {:>2}  held {:>2}  {:?}",
      step.stride,
      outcomes
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join(","),
      step.last_agreed_seconds,
      step.confirmed.len(),
      step.held_back.len(),
      step.confirmed.join(" "),
    );
    steps.push(step);
  }
  // Read before `finalize` consumes the driver: the floor the prefix check needs
  // is this run's own agreement width, not a literal.
  let agreement_count_needed = streamer.agreement().agreement_count_needed();
  let confirmed = normalized(streamer.finalize().text());
  println!("[confirmed] {confirmed:?}");

  // ── Portable, asserted on EVERY host, and asserted FIRST so a real
  // confirmation regression is never reported from behind a host-scoped gate.
  if let Err(why) = check_monotone_confirmation(&steps) {
    panic!("streaming confirmation was not monotone: {why}");
  }
  if let Err(why) = check_confirmed_is_prefix_of_batch(&confirmed, &batch, agreement_count_needed) {
    panic!("streaming confirmation is not a prefix of the batch transcript: {why}");
  }

  // ── Measured, asserted only on the host class that produced it.
  gate.check_holds(
    "confirmed stream reaches the clip's canonical phrase",
    confirmed.contains(CANONICAL_PHRASE),
    &format!("contains {CANONICAL_PHRASE:?}"),
    &format!(
      "the confirmed stream never reached the canonical phrase on the host that \
       characterized it.\n  confirmed : {confirmed:?}\n  batch     : {batch:?}\n\
       The portable properties above already passed, so the confirmation is still a \
       faithful\ntruncation — what changed is HOW FAR it got before the timestamp-mass \
       gate stalled it\n(see this file's module docs). Do NOT weaken, widen or delete \
       the phrase to make this\npass; re-measure and re-record CHARACTERIZED_ON, or \
       diagnose the stall."
    ),
  );
}

// ---------------------------------------------------------------------
// Falsifiers for the portable predicates — hermetic, NOT model-gated
// ---------------------------------------------------------------------

/// Snapshots from normalized word lists, for the hermetic cases below.
fn step(stride: usize, watermark: f32, confirmed: &[&str], held_back: &[&str]) -> Confirmation {
  Confirmation {
    stride,
    last_agreed_seconds: watermark,
    confirmed: confirmed.iter().map(|w| (*w).to_string()).collect(),
    held_back: held_back.iter().map(|w| (*w).to_string()).collect(),
  }
}

const BATCH: &str = "and so my fellow americans ask not what your country can do for you";

/// A genuine truncation is what LocalAgreement-2 is FOR, and must pass.
#[test]
fn a_truncated_confirmation_is_accepted() {
  assert_eq!(
    check_confirmed_is_prefix_of_batch("and so my fellow americans", BATCH, 2),
    Ok(())
  );
  // The degenerate-but-legal ends: exactly the agreement width, and the whole
  // transcript.
  assert_eq!(
    check_confirmed_is_prefix_of_batch("and so", BATCH, 2),
    Ok(())
  );
  assert_eq!(check_confirmed_is_prefix_of_batch(BATCH, BATCH, 2), Ok(()));
}

/// THE falsifier for the prefix property: a confirmation that REWRITES a word
/// instead of stopping short must red, and must name the word.
#[test]
fn a_rewritten_word_is_not_a_truncation() {
  let why = check_confirmed_is_prefix_of_batch("and so my fellow australians", BATCH, 2)
    .expect_err("a rewritten word must not pass the prefix check");
  assert!(why.contains("REWRITES"), "{why}");
  assert!(why.contains("first divergence at word 4"), "{why}");
  assert!(why.contains("australians"), "{why}");

  // A word DROPPED from the middle is a rewrite too: everything after it shifts
  // left, so position 3 no longer holds `fellow`.
  let why = check_confirmed_is_prefix_of_batch("and so my americans", BATCH, 2)
    .expect_err("a dropped middle word must not pass the prefix check");
  assert!(why.contains("first divergence at word 3"), "{why}");
}

/// A confirmation that runs PAST the batch transcript is not a truncation
/// either — the shape a word confirmed TWICE produces, whether from
/// `LocalAgreement::finalize`'s double fold with its `holdback_superseded` guard
/// not doing its job, or from the re-admission strip letting an
/// already-confirmed word back into a hypothesis.
#[test]
fn a_confirmation_longer_than_the_batch_is_not_a_truncation() {
  let why = check_confirmed_is_prefix_of_batch(&format!("{BATCH} for you"), BATCH, 2)
    .expect_err("a confirmation longer than the batch must not pass");
  assert!(why.contains("LONGER"), "{why}");
  assert!(why.contains("finalize"), "{why}");
}

/// An EMPTY confirmation is a prefix of everything, so without a floor the
/// prefix check would be decorative. It must red instead.
#[test]
fn an_empty_confirmation_is_not_evidence() {
  let why = check_confirmed_is_prefix_of_batch("", BATCH, 2)
    .expect_err("an empty confirmation must not pass the prefix check");
  assert!(why.contains("confirmed 0 word(s)"), "{why}");
  assert!(why.contains("stall"), "{why}");

  // One word short of a completed agreement round is refused for the same
  // reason — the floor is the run's own `agreement_count_needed`, not zero.
  let why = check_confirmed_is_prefix_of_batch("and", BATCH, 2)
    .expect_err("a sub-agreement-width confirmation must not pass");
  assert!(why.contains("confirmed 1 word(s)"), "{why}");
}

/// A growing confirmation with a rising watermark is the healthy shape.
#[test]
fn a_growing_confirmation_is_accepted() {
  let steps = [
    step(1, 0.0, &[], &[]),
    step(2, 0.0, &[], &["and", "so"]),
    step(3, 0.62, &["and"], &["so", "my"]),
    step(4, 0.94, &["and", "so"], &["my", "fellow"]),
  ];
  assert_eq!(check_monotone_confirmation(&steps), Ok(()));
}

/// THE falsifier for monotonicity: a confirmed word rewritten by a later stride.
#[test]
fn a_retracted_confirmed_word_reds_monotonicity() {
  let steps = [
    step(1, 0.62, &["and", "so"], &["my", "fellow"]),
    step(2, 0.94, &["and", "then"], &["my", "fellow"]),
  ];
  let why = check_monotone_confirmation(&steps)
    .expect_err("a rewritten confirmed word must red monotonicity");
  assert!(why.contains("CONFIRMED word was rewritten"), "{why}");
  assert!(why.contains("at index 1"), "{why}");
  assert!(why.contains("then"), "{why}");
}

/// A confirmed list that gets SHORTER is a retraction with its own message.
#[test]
fn a_shrinking_confirmed_list_reds_monotonicity() {
  let steps = [
    step(1, 0.62, &["and", "so", "my"], &["fellow", "americans"]),
    step(2, 0.94, &["and", "so"], &["my", "fellow"]),
  ];
  let why = check_monotone_confirmation(&steps).expect_err("a shrinking confirmed list must red");
  assert!(why.contains("SHRANK"), "{why}");
  assert!(why.contains("3 -> 2 words"), "{why}");
}

/// A watermark that moves back re-opens settled text.
#[test]
fn a_receding_watermark_reds_monotonicity() {
  let steps = [
    step(1, 1.50, &["and", "so"], &["my", "fellow"]),
    step(2, 0.94, &["and", "so"], &["my", "fellow"]),
  ];
  let why = check_monotone_confirmation(&steps).expect_err("a receding watermark must red");
  assert!(why.contains("RECEDED"), "{why}");
  assert!(why.contains("1.5 s -> 0.94 s"), "{why}");
}

/// A run that never confirms anything satisfies every monotonicity check
/// trivially, so the predicate must refuse it rather than report a green.
#[test]
fn a_run_that_confirms_nothing_is_not_monotone_evidence() {
  let steps = [step(1, 0.0, &[], &[]), step(2, 0.0, &[], &[])];
  let why = check_monotone_confirmation(&steps).expect_err("a run with no confirmation must red");
  assert!(why.contains("confirmed and held back NOTHING"), "{why}");

  let why = check_monotone_confirmation(&[]).expect_err("a run with no pushes at all must red");
  assert!(why.contains("no pushes were observed"), "{why}");
}

// ---------------------------------------------------------------------
// The phrase gate's three host verdicts — hermetic, NOT model-gated
// ---------------------------------------------------------------------

/// A synthetic running host, deliberately not this machine's, so these run the
/// same everywhere.
fn synthetic_running_host() -> common::HostClass {
  common::HostClass {
    os_build: "25F71".to_string(),
    os_product_version: "26.5".to_string(),
    chip: "Apple M9 Ultra".to_string(),
    arch: "arm64".to_string(),
  }
}

/// The recorded constant matching [`synthetic_running_host`].
const SYNTHETIC_SAME_HOST: common::CharacterizedHost = common::CharacterizedHost {
  os_build: "25F71",
  os_product_version: "26.5",
  chip: "Apple M9 Ultra",
  arch: "arm64",
};

/// A recorded constant for a different chip — the axis the Neural Engine's fp16
/// arithmetic actually varies along, and the one separating this machine from
/// the CI runner.
const SYNTHETIC_OTHER_HOST: common::CharacterizedHost = common::CharacterizedHost {
  os_build: "24G720",
  os_product_version: "15.7.7",
  chip: "Apple M1 (Virtual)",
  arch: "arm64",
};

fn phrase_gate(recorded: Option<common::CharacterizedHost>) -> common::BandGate {
  common::BandGate::open_with(
    "whisper streaming confirmed phrase (hermetic)",
    recorded,
    synthetic_running_host(),
    recharacterize_command(),
  )
}

/// Calls the phrase gate exactly as the model-gated test does, catching the
/// panic an ARMED failure raises.
fn drive_phrase_gate(gate: &common::BandGate, present: bool) -> Result<String, ()> {
  catch_unwind(AssertUnwindSafe(|| {
    gate.check_holds(
      "confirmed stream reaches the clip's canonical phrase",
      present,
      &format!("contains {CANONICAL_PHRASE:?}"),
      "the confirmed stream never reached the canonical phrase",
    )
  }))
  .map_err(|_| ())
}

/// ARMED: on the host the phrase was measured on the gate is exactly as strict
/// as the unconditional assertion it replaced.
#[test]
fn an_armed_phrase_gate_is_strict() {
  let gate = phrase_gate(Some(SYNTHETIC_SAME_HOST));
  assert_eq!(gate.verdict(), common::BandVerdict::Armed);
  assert!(gate.armed());
  assert!(gate.banner().contains("ARE ASSERTED"), "{}", gate.banner());

  let line = drive_phrase_gate(&gate, true).expect("a present phrase must not panic");
  assert!(line.contains("[ASSERTED]"), "{line}");
  assert!(!line.contains("BAND NOT ASSERTED"), "{line}");

  assert!(
    drive_phrase_gate(&gate, false).is_err(),
    "an ARMED gate must still red on a missing phrase — otherwise host-scoping the \
     phrase would have deleted it"
  );
}

/// FOREIGN: the CI runner's case. Reported with both hosts named, never
/// asserted — arming a phrase on a machine known not to produce it would assert
/// something false.
#[test]
fn a_foreign_phrase_gate_reports_and_cannot_red() {
  let gate = phrase_gate(Some(SYNTHETIC_OTHER_HOST));
  assert_eq!(gate.verdict(), common::BandVerdict::Foreign);
  assert!(!gate.armed());

  let banner = gate.banner();
  assert!(banner.contains("NOT ASSERTED"), "{banner}");
  assert!(banner.contains("Apple M1 (Virtual)"), "{banner}");
  assert!(banner.contains("Apple M9 Ultra"), "{banner}");

  let line = drive_phrase_gate(&gate, false).expect("a foreign gate must never panic");
  assert!(line.contains("MISSING"), "{line}");
  assert!(line.contains("BAND NOT ASSERTED"), "{line}");
  assert!(
    line.contains("characterized on a different host class"),
    "{line}"
  );
}

/// UNRECORDED: no host in source, so nothing is known about which machine
/// produced the phrase. Reported, never asserted.
#[test]
fn an_unrecorded_phrase_gate_reports_and_cannot_red() {
  let gate = phrase_gate(None);
  assert_eq!(gate.verdict(), common::BandVerdict::Unrecorded);
  assert!(!gate.armed());
  assert!(
    gate.banner().contains("CHARACTERIZED_ON = None"),
    "{}",
    gate.banner()
  );

  let line = drive_phrase_gate(&gate, false).expect("an unrecorded gate must never panic");
  assert!(line.contains("MISSING"), "{line}");
  assert!(line.contains("no characterization host recorded"), "{line}");
}

/// The phrase is the ONLY thing behind the host gate: the portable properties
/// are asserted bare, so no verdict can silence them. Checked against this
/// file's real source, the way siglip's `band_provenance` checks its own.
#[test]
fn only_the_phrase_is_host_scoped() {
  let source = std::fs::read_to_string(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/whisper/streaming.rs"
  ))
  .expect("this test file is readable");

  // Scoped to the model-gated test's own body, so this scan cannot match itself
  // or the hermetic verdict tests below.
  let body = source
    .split_once("fn jfk_simulated_stream_confirms_the_transcript(")
    .expect("the model-gated test is still here")
    .1
    .split_once("// Falsifiers for the portable predicates")
    .expect("the falsifier section still follows it")
    .0;

  // Exactly one measurement rides the gate, and it is the phrase. A second
  // `gate.check_` would mean a portable property had been moved behind a
  // verdict that can switch it off.
  let gated: Vec<&str> = body
    .lines()
    .filter(|line| line.contains("gate.check_"))
    .collect();
  assert_eq!(
    gated.len(),
    1,
    "exactly one measurement may ride the host gate, and it must be the phrase: {gated:?}"
  );
  assert!(gated[0].contains("check_holds"), "{:?}", gated[0]);

  // Both portable properties are called bare, so no verdict can silence them.
  for portable in [
    "check_monotone_confirmation(&steps)",
    "check_confirmed_is_prefix_of_batch(&confirmed, &batch, agreement_count_needed)",
  ] {
    assert!(
      body.contains(portable),
      "portable property `{portable}` is no longer asserted in the model-gated test"
    );
  }

  // Host-scoping moved the phrase; it did not relax it.
  assert!(
    source.contains("const CANONICAL_PHRASE: &str = \"ask not what your country can do for you\";"),
    "the phrase must not be weakened, widened or deleted — it is host-scoped, not relaxed"
  );
}
