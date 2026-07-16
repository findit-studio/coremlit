//! **The public `prepare` → `Encoder` → `finish` composition path**, driven
//! through asry's and alignkit's public surface only.
//!
//! asry's public [`EmissionsAligner::prepare`](asry::emissions::EmissionsAligner::prepare)
//! hands the caller a receptive-field-padded
//! [`encoder_input`](asry::emissions::PreparedChunk::encoder_input) buffer to feed
//! their own encoder, plus the true pre-pad
//! [`real_samples`](asry::emissions::PreparedChunk::real_samples).
//! [`EncoderInput::from_prepared`] reads BOTH off the one (unforgeable)
//! [`PreparedChunk`](asry::emissions::PreparedChunk), so the frame count is
//! truncated from asry's own authoritative extent — the same extent `finish`
//! validates against.
//!
//! Two things are pinned here, end-to-end on the supported public route:
//!
//! - [`public_prepared_composition_keeps_only_real_frames`]: a 200-sample chunk
//!   (padded to the 400-sample receptive field) keeps exactly **one** emission
//!   frame — the wav2vec2 conv stack's output for a sub-receptive-field input.
//! - [`public_prepared_composition_641_abc_has_no_alignment_path`]: the codex-fence
//!   regression. 641 real samples truncate to **one** frame
//!   (`floor((641 − 400) / 320) + 1`), and one frame cannot carry the three
//!   distinct tokens of `ABC`, so `finish` returns `NoAlignmentPath` — where the
//!   old `ceil(641/320) = 3` gave three phantom, padding-derived frames and a
//!   plausible-but-nonexistent alignment. asry's `chunk_extent ± 2·hop` stride
//!   check (`3×320 = 960` inside `641 ± 640`) is too loose to catch it, so this
//!   end-to-end test is the guard.
//!
//! Model-gated like the other end-to-end tests: `#[ignore]` is the only gate, a
//! missing model is a hard failure (see `tests/align_chunk.rs`'s module doc).

mod common;

use core::sync::atomic::AtomicBool;

use alignkit::{
  ANALYSIS_TIMEBASE, EnglishNormalizer, Lang, OutputClock, SpeechSpans,
  encode::{Encoder, EncoderInput},
  vocab,
};
use asry::emissions::{EmissionsAligner, EmissionsError};

/// Builds asry's public [`EmissionsAligner`] the way alignkit's own `Aligner`
/// wires its seam — bundled 29-class chordai tokenizer + the MANDATORY explicit
/// blank id (`vocab::BLANK_ID`, since that vocab has no `<pad>`/`[PAD]`/`<blank>`
/// entry for asry's auto-detect) — but reachable here through asry's and
/// alignkit's PUBLIC surface only. That is the point: this is the supported
/// external composition, not the crate-internal `Aligner::align_chunk` one.
fn en_emissions_aligner() -> EmissionsAligner {
  EmissionsAligner::builder(Lang::En, vocab::tokenizer_json_bytes())
    .normalizer(Box::new(EnglishNormalizer::new()))
    .blank_token_id(vocab::BLANK_ID)
    .build()
    .expect("build the En EmissionsAligner from alignkit's bundled tokenizer")
}

/// The composition path keeps exactly the real audio's frame count. An external
/// caller composes `EmissionsAligner::prepare` → alignkit [`Encoder`] by hand;
/// `prepare` on 200 real samples returns a
/// [`PreparedChunk`](asry::emissions::PreparedChunk) whose `encoder_input()` is
/// those 200 samples zero-padded to wav2vec2's 400-sample receptive field.
///
/// [`EncoderInput::from_prepared`] reads the true pre-pad length (200) off the
/// chunk, and the conv-geometry truncation keeps the single receptive-field
/// frame: `floor((200.max(400) − 400) / 320) + 1 = 1`.
///
/// # The door choice is now benign for the count
///
/// Before the conv-geometry fix, feeding the padded buffer through
/// [`EncoderInput::from_samples`] recorded 400 padded samples as real and kept
/// `ceil(400/320) = 2` frames where `ceil(200/320) = 1` belonged — the F1
/// divergence. With the corrected truncation both doors now keep **1** frame for
/// any sub-receptive-field chunk (the conv stack pads to the receptive field
/// either way), so this specific slip no longer changes the count. `from_prepared`
/// is still the correct, self-documenting door — it reads the honest pre-pad
/// length and stays right for the general case — but the count no longer depends
/// on the choice here. Both are asserted, so a regression in EITHER door is caught.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn public_prepared_composition_keeps_only_real_frames() {
  let aligner = en_emissions_aligner();
  let encoder = Encoder::from_file(common::model_path())
    .expect("load base960h_aligner.mlmodelc (set ALIGNKIT_TEST_MODELS)");

  // 200 real samples of ACTUAL speech (valid, finite, log-prob-producing audio,
  // so `emissions()` returns Ok at all). The content is irrelevant to the FRAME
  // COUNT under test — only `real_samples` drives truncation.
  let jfk = common::load_wav_mono_f32(&common::jfk_wav_path());
  let samples = &jfk[80_000..80_200];
  assert_eq!(
    samples.len(),
    200,
    "the F1 case is exactly 200 real samples"
  );

  let abort = AtomicBool::new(false);
  let prepared = aligner
    .prepare(samples, &SpeechSpans::all_speech(), "test", &[], &abort)
    .expect("prepare 200 real samples with alignable text");
  assert!(
    !prepared.is_trivial(),
    "text `test` must tokenize to alignable tokens, or there is no encoder buffer to test"
  );
  assert_eq!(
    prepared.encoder_input().len(),
    400,
    "asry pads 200 real samples up to the 400-sample receptive field"
  );

  // The supported door: real length read off the chunk → one receptive-field frame.
  let correct = encoder
    .emissions(EncoderInput::from_prepared(&prepared).expect("from_prepared geometry is valid"))
    .expect("emissions on the prepared chunk");
  assert_eq!(
    correct.frames(),
    1,
    "from_prepared keeps the single receptive-field frame: floor((200.max(400) - 400)/320) + 1 = 1"
  );

  // The old raw-door composition: feed the padded buffer as raw audio, recording
  // its 400 samples as real. The ceil formula kept 2 frames here (the F1 bug); the
  // conv formula keeps the SAME single frame, because 400 samples is exactly one
  // receptive field — the door choice no longer moves the count for a
  // sub-receptive-field chunk.
  let via_raw_door = encoder
    .emissions(
      EncoderInput::from_samples(prepared.encoder_input()).expect("from_samples geometry is valid"),
    )
    .expect("emissions on the padded buffer fed as raw audio");
  assert_eq!(
    via_raw_door.frames(),
    1,
    "conv geometry keeps one frame for the 400-sample padded buffer too (was ceil(400/320) = 2)"
  );
}

/// **The codex-fence regression, on the public composition path.** 641 real
/// samples carrying three distinct tokens (`ABC`) must yield `NoAlignmentPath`
/// from `finish` — one frame cannot carry three tokens.
///
/// 641 real samples truncate to `floor((641 − 400) / 320) + 1 = 1` emission
/// frame — asry's own ONNX encoder produces that same one frame. A CTC forced
/// aligner needs at least one frame per distinct token, so three distinct tokens
/// against one frame has no valid path and `finish` returns
/// [`EmissionsError::NoAlignmentPath`].
///
/// The old `ceil(641/320) = 3` kept three phantom, padding-derived frames, into
/// which the trellis threaded a plausible-but-nonexistent `A B C` alignment and
/// returned `Ok` — asry's `chunk_extent ± 2·hop` stride check (`3×320 = 960`
/// inside `641 ± 640`) is too loose to catch it. This is the standing mutation
/// proof: revert `truncated_frame_count` to `ceil` and BOTH the `frames() == 1`
/// check (it becomes 3) and the `NoAlignmentPath` check (`finish` returns `Ok`)
/// fail here.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn public_prepared_composition_641_abc_has_no_alignment_path() {
  let aligner = en_emissions_aligner();
  let encoder = Encoder::from_file(common::model_path())
    .expect("load base960h_aligner.mlmodelc (set ALIGNKIT_TEST_MODELS)");

  // 641 real samples of ACTUAL speech, and a transcript of three distinct in-vocab
  // letters. The frame COUNT (1) forces NoAlignmentPath, independent of the
  // emission VALUES — but the audio must be valid log-prob-producing input for
  // `emissions()` to return Ok at all.
  let jfk = common::load_wav_mono_f32(&common::jfk_wav_path());
  let samples = &jfk[80_000..80_641];
  assert_eq!(
    samples.len(),
    641,
    "the fence case is exactly 641 real samples"
  );
  let text = "ABC";
  assert!(
    aligner.detect_oov(text).expect("detect_oov").is_empty(),
    "A, B, C must be in-vocab, or the OOV path — not the frame count — would drive the result"
  );

  let abort = AtomicBool::new(false);
  let prepared = aligner
    .prepare(samples, &SpeechSpans::all_speech(), text, &[], &abort)
    .expect("prepare 641 real samples with `ABC`");
  assert!(
    !prepared.is_trivial(),
    "`ABC` must tokenize to three alignable tokens"
  );

  // One real frame: the codex-fence count. The old ceil kept three.
  let emissions = encoder
    .emissions(EncoderInput::from_prepared(&prepared).expect("from_prepared geometry is valid"))
    .expect("emissions on the 641-sample chunk");
  assert_eq!(
    emissions.frames(),
    1,
    "641 real samples truncate to one frame: floor((641 - 400)/320) + 1 = 1 (was ceil → 3)"
  );

  // finish: three distinct tokens cannot be aligned to one frame → NoAlignmentPath.
  let clock = OutputClock::new(0, ANALYSIS_TIMEBASE, 0).expect("clock construction");
  let err = aligner
    .finish(prepared, &emissions, clock, &abort)
    .expect_err("one frame cannot carry three distinct tokens; finish must return NoAlignmentPath");
  assert!(
    matches!(err, EmissionsError::NoAlignmentPath(_)),
    "expected NoAlignmentPath (the fence's reference outcome), got {err:?}"
  );
}
