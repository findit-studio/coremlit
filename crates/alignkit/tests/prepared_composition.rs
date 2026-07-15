//! **F1 regression** — the PUBLIC `prepare` → `Encoder` → `finish` composition
//! path has a correct prepared-buffer door.
//!
//! asry's public [`EmissionsAligner::prepare`](asry::emissions::EmissionsAligner::prepare)
//! hands the caller a receptive-field-padded
//! [`encoder_input`](asry::emissions::PreparedChunk::encoder_input) buffer to
//! feed their own encoder. An external caller composing that seam with alignkit's
//! [`Encoder`] used to have only [`EncoderInput::from_samples`], which records the
//! PADDED buffer length as real — so 200 real samples zero-padded to 400 kept
//! `ceil(400/320) = 2` emission frames where `ceil(200/320) = 1` belongs. asry's
//! `chunk_extent ± 2·hop` stride slack accepts the two-frame lie, yielding a
//! plausible but wrong alignment with no error.
//!
//! The fix is [`EncoderInput::from_prepared`], which reads BOTH the padded buffer
//! and the true pre-pad real length off the one (unforgeable)
//! [`PreparedChunk`](asry::emissions::PreparedChunk) — asry's
//! [`real_samples`](asry::emissions::PreparedChunk::real_samples), newly public
//! for exactly this composition. This test drives the supported public route and
//! pins `emissions.frames() == 1`, and — as the mutation proof — pins that the OLD
//! composition (`from_samples(prepared.encoder_input())`) keeps 2.
//!
//! Model-gated like the other end-to-end tests: `#[ignore]` is the only gate, a
//! missing model is a hard failure (see `tests/align_chunk.rs`'s module doc).

mod common;

use core::sync::atomic::AtomicBool;

use alignkit::{
  EnglishNormalizer, Lang, SpeechSpans,
  encode::{Encoder, EncoderInput},
  vocab,
};
use asry::emissions::EmissionsAligner;

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

/// The F1 regression, end-to-end on the supported public path. An external caller
/// composes `EmissionsAligner::prepare` → alignkit [`Encoder`] by hand; `prepare`
/// on 200 real samples returns a [`PreparedChunk`](asry::emissions::PreparedChunk)
/// whose `encoder_input()` is those 200 samples zero-padded to wav2vec2's
/// 400-sample receptive field.
///
/// [`EncoderInput::from_prepared`] reads the true pre-pad length (200) off the
/// chunk, so the emissions keep `ceil(200/320) = 1` frame. The OLD composition —
/// the only public door before this fix — could feed the padded buffer only
/// through [`EncoderInput::from_samples`], recording 400 padded samples as real
/// and keeping `ceil(400/320) = 2` frames. Both counts are asserted, so the wrong
/// one is the standing mutation proof that the prepared door is load-bearing.
#[test]
#[ignore = "requires local alignkit models (ALIGNKIT_TEST_MODELS)"]
fn public_prepared_composition_keeps_only_real_frames() {
  let aligner = en_emissions_aligner();
  let encoder = Encoder::from_file(common::model_path())
    .expect("load base960h_aligner.mlmodelc (set ALIGNKIT_TEST_MODELS)");

  // 200 real samples of ACTUAL speech: non-constant, so asry's zero-mean /
  // unit-variance normalize is well-defined and the emissions stay in the
  // log-prob domain. The content is irrelevant to the FRAME COUNT under test —
  // only `real_samples` drives truncation — but it must be valid, finite,
  // log-prob-producing audio for `emissions()` to return Ok at all.
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

  // The supported door: real length read off the chunk → ceil(200/320) = 1.
  let correct = encoder
    .emissions(EncoderInput::from_prepared(&prepared).expect("from_prepared geometry is valid"))
    .expect("emissions on the prepared chunk");
  assert_eq!(
    correct.frames(),
    1,
    "from_prepared keeps only the real frame: ceil(200/320) = 1"
  );

  // The OLD composition (the mutation): feed the padded buffer as raw audio, so
  // its 400 samples are recorded as real → ceil(400/320) = 2 frames. This is the
  // plausible-but-wrong alignment the fix closes; pinning it proves the door above
  // is what makes the difference.
  let wrong = encoder
    .emissions(
      EncoderInput::from_samples(prepared.encoder_input()).expect("from_samples geometry is valid"),
    )
    .expect("emissions on the padded buffer fed as raw audio");
  assert_eq!(
    wrong.frames(),
    2,
    "from_samples on the padded buffer keeps a padded-tail frame: ceil(400/320) = 2"
  );

  assert_ne!(
    correct.frames(),
    wrong.frames(),
    "the prepared door and the raw door disagree on the padded buffer — that gap is F1"
  );
}
