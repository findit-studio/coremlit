use super::*;
use coremlit::ComputeUnits;

// =====================================================================
// Hermetic: Source (rust-type-conventions vocabulary enum)
// =====================================================================

#[test]
fn source_default_is_fluid_audio() {
  assert_eq!(Source::default(), Source::FluidAudio);
  assert_eq!(DEFAULT_SOURCE, Source::FluidAudio);
}

#[test]
fn source_variants_are_exhaustively_matchable() {
  // No wildcard arm: this only compiles if `Source` still has exactly
  // these two variants — pins the enum's shape so a future variant
  // addition must touch this match, and keeps `Argmax` genuinely
  // matchable rather than silently absorbed by a catch-all (module doc's
  // rationale for NOT marking `Source` `#[non_exhaustive]`).
  let cases = [
    (Source::FluidAudio, "fluid_audio"),
    (Source::Argmax, "argmax"),
  ];
  for (source, expected) in cases {
    let label = match source {
      Source::FluidAudio => "fluid_audio",
      Source::Argmax => "argmax",
    };
    assert_eq!(label, expected);
  }
}

#[cfg(feature = "serde")]
#[test]
fn source_serde_wire_values_are_snake_case() {
  assert_eq!(
    serde_json::to_string(&Source::FluidAudio).unwrap(),
    "\"fluid_audio\""
  );
  assert_eq!(
    serde_json::to_string(&Source::Argmax).unwrap(),
    "\"argmax\""
  );
}

#[cfg(feature = "serde")]
#[test]
fn source_serde_round_trips() {
  for source in [Source::FluidAudio, Source::Argmax] {
    let json = serde_json::to_string(&source).unwrap();
    let back: Source = serde_json::from_str(&json).unwrap();
    assert_eq!(back, source);
  }
}

// =====================================================================
// Hermetic: AnySource (the dispatcher)
// =====================================================================

// `AnySource`'s one-variant-per-`Source` correspondence needs no test of its
// own: all three of its matches (`load`, `source`, and the `ModelSource`
// impl) are exhaustive with no wildcard arm, so a `Source` variant with no
// `AnySource` counterpart — the only way one source could silently route to
// another — fails to COMPILE. The two properties a test CAN add are the
// no-fallback error path (below) and real dispatch
// (`any_source_load_dispatches_by_source`, model-gated).

/// Loading `Source::Argmax` from a directory that holds only the FluidAudio
/// artifacts must FAIL, not silently fall back to `FluidAudioSource` — the
/// central honesty property of the dispatcher. (Hermetic: the load fails on
/// a nonexistent path, no models needed.)
#[test]
fn any_source_argmax_does_not_fall_back_to_fluid_audio() {
  let nowhere = std::path::Path::new("/nonexistent-speakerkit-models");
  let got = AnySource::load(nowhere, Options::new().with_source(Source::Argmax));
  assert!(
    matches!(got, Err(crate::error::ModelError::Load(_))),
    "a missing argmax model must surface as a load error, never a \
     FluidAudio source; got {got:?}"
  );
  // And the FluidAudio arm fails independently on the same missing path.
  let got = AnySource::load(nowhere, Options::new().with_source(Source::FluidAudio));
  assert!(matches!(got, Err(crate::error::ModelError::Load(_))));
}

/// [`AnySource::load`] builds the source [`Options::source`] names — and
/// dispatches `extract` to it.
#[test]
#[ignore = "requires local argmax + speakerkit models (both env vars)"]
fn any_source_load_dispatches_by_source() {
  let samples = load_ted_head();

  let fluid = AnySource::load(models_dir(), Options::new().with_source(Source::FluidAudio))
    .expect("load FluidAudio via the dispatcher");
  assert_eq!(fluid.source(), Source::FluidAudio);
  assert!(matches!(fluid, AnySource::FluidAudio(_)));
  let fluid_out = fluid.extract(&samples).expect("dispatched extract");

  let argmax_dir = std::env::var_os("ARGMAX_TEST_MODELS").map_or_else(
    || {
      std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("argmax-speakerkit")
    },
    std::path::PathBuf::from,
  );
  let argmax = AnySource::load(argmax_dir, Options::new().with_source(Source::Argmax))
    .expect("load Argmax via the dispatcher");
  assert_eq!(argmax.source(), Source::Argmax);
  assert!(matches!(argmax, AnySource::Argmax(_)));
  let argmax_out = argmax.extract(&samples).expect("dispatched extract");

  // THAT each call ran its own source is already proven above, by the
  // compiler: an `AnySource::Argmax` can only dispatch to `ArgmaxSource`
  // (the `ModelSource` impl's match is exhaustive and wildcard-free), and
  // `matches!` pinned which variant was built. No output comparison is
  // needed for that, and none would be sound as a proxy — see below.
  //
  // Geometry must agree (the grid theorem in `argmax`'s module doc).
  assert_eq!(argmax_out.num_chunks(), fluid_out.num_chunks());
  assert_eq!(
    argmax_out.num_output_frames(),
    fluid_out.num_output_frames()
  );
  assert_eq!(argmax_out.chunks_sw(), fluid_out.chunks_sw());

  // VALUES are NOT required to agree — the two decodes are independent (spec
  // §4) — but nor are they required to DIFFER, and asserting they must would
  // be wrong: on this short, clean, single-speaker clip the two hard decodes
  // in fact produce IDENTICAL `segmentations`, which is unsurprising (same
  // underlying pyannote model, two conversions) and reassuring. Only the
  // embeddings must differ, because the two sources run entirely different
  // embedding networks (WeSpeaker vs argmax's `SpeakerEmbedder`) — and both
  // are non-trivial here, so this cannot pass vacuously on two zero buffers.
  assert!(fluid_out.raw_embeddings().iter().any(|&v| v != 0.0));
  assert!(argmax_out.raw_embeddings().iter().any(|&v| v != 0.0));
  assert_ne!(
    argmax_out.raw_embeddings(),
    fluid_out.raw_embeddings(),
    "two different embedding networks cannot produce bit-identical embeddings"
  );
}

// =====================================================================
// Model-gated (all #[ignore]): requires local speakerkit models
// (SPEAKERKIT_TEST_MODELS or Models/speakerkit/) plus the cross-crate
// ted_60.wav fixture. Loader/path helpers duplicated in miniature — same
// reason as crate::extract::tests, crate::embed::tests, and
// crate::segment::tests: unit tests under `src/` cannot import the
// separate `tests/` integration-test crate.
// =====================================================================

fn models_dir() -> std::path::PathBuf {
  std::env::var_os("SPEAKERKIT_TEST_MODELS").map_or_else(
    || {
      std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models")
        .join("speakerkit")
    },
    std::path::PathBuf::from,
  )
}

fn load_seg_model() -> SegmentModel {
  // CpuOnly for determinism, matching crate::extract::tests::load_seg_model
  // and every other model-gated loader in this crate.
  SegmentModel::from_file_with(
    models_dir().join("pyannote_segmentation.mlmodelc"),
    crate::segment::SegmentModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load pyannote_segmentation.mlmodelc")
}

fn load_embed_model() -> EmbedModel {
  EmbedModel::from_file_with(
    models_dir().join("wespeaker_v2.mlmodelc"),
    crate::embed::EmbedModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load wespeaker_v2.mlmodelc")
}

/// The first 2 s (32_000 samples at 16 kHz) of the cross-crate `ted_60.wav`
/// fixture (see `crate::extract::tests::load_ted_60` for the full-clip
/// loader) — long enough to be a real, non-degenerate segmentation chunk,
/// short enough (`<= SEG_CHUNK_SAMPLES`) that `crate::window::chunk_starts`
/// always yields exactly one chunk, keeping these equivalence tests fast.
fn load_ted_head() -> Vec<f32> {
  let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("../whisperkit/tests/fixtures/audio/ted_60.wav");
  let mut reader = hound::WavReader::open(&path).expect("ted_60.wav opens");
  let spec = reader.spec();
  assert_eq!(spec.channels, 1, "fixture must be mono");
  assert_eq!(spec.sample_rate, 16_000, "fixture must be 16 kHz");
  assert_eq!(spec.sample_format, hound::SampleFormat::Int);
  let samples: Vec<f32> = reader
    .samples::<i16>()
    .take(32_000)
    .map(|s| f32::from(s.expect("valid sample")) / 32_768.0)
    .collect();
  assert_eq!(samples.len(), 32_000, "ted_60.wav has at least 2 s");
  samples
}

/// THE equivalence test (brief step 1): a [`FluidAudioSource`] built from
/// the two models must produce the SAME [`Extraction`] as
/// [`Extractor::extract`] on identical input and default [`Options`].
/// Loads each model twice (once per call path) since [`SegmentModel`]/
/// [`EmbedModel`] are not `Clone` — `FluidAudioSource` owns its pair, so
/// there is no way to share one loaded instance across both call paths.
#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn fluid_audio_source_matches_extractor_default_options() {
  let samples = load_ted_head();

  let seg_a = load_seg_model();
  let embed_a = load_embed_model();
  let want = Extractor::new()
    .extract(&seg_a, &embed_a, &samples)
    .expect("Extractor::extract on the ted head");

  let seg_b = load_seg_model();
  let embed_b = load_embed_model();
  let got = FluidAudioSource::new(seg_b, embed_b)
    .extract(&samples)
    .expect("FluidAudioSource::extract on the ted head");

  assert_eq!(
    got, want,
    "FluidAudioSource::extract must byte-match Extractor::extract"
  );
  // Named-accessor comparisons too (brief: "byte-equal accessors"), not
  // just the whole-struct PartialEq above.
  assert_eq!(got.raw_embeddings(), want.raw_embeddings());
  assert_eq!(got.segmentations(), want.segmentations());
  assert_eq!(got.count(), want.count());
  assert_eq!(got.num_chunks(), want.num_chunks());
  assert_eq!(got.num_frames_per_chunk(), want.num_frames_per_chunk());
  assert_eq!(got.num_output_frames(), want.num_output_frames());
}

/// Same equivalence claim, but with `Options` that diverge from
/// `Options::default()` on both fields `Extractor::extract` actually
/// reads (`window.onset`, `window.step_samples`) — catches a regression
/// where `FluidAudioSource::extract` drops `self.options` and calls
/// `Extractor::new()` instead of `Extractor::with_options(self.options)`
/// (a default-options-only test cannot distinguish those two).
#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn fluid_audio_source_matches_extractor_custom_options() {
  let options = Options::new().with_window(
    crate::window::WindowOptions::new()
      .with_onset(0.3)
      .with_step_samples(8_000),
  );
  let samples = load_ted_head();

  let seg_a = load_seg_model();
  let embed_a = load_embed_model();
  let want = Extractor::with_options(options)
    .extract(&seg_a, &embed_a, &samples)
    .expect("Extractor::extract with custom options");

  let seg_b = load_seg_model();
  let embed_b = load_embed_model();
  let got = FluidAudioSource::with_options(seg_b, embed_b, options)
    .extract(&samples)
    .expect("FluidAudioSource::extract with custom options");

  assert_eq!(
    got, want,
    "FluidAudioSource::extract must thread self.options through, not just self.seg/self.embed"
  );
}

/// Error paths must match too, not just the success path: both call paths
/// reject empty `samples` identically. Model-gated only because
/// `FluidAudioSource::new`/`Extractor::extract` both require loaded
/// models to construct/call, mirroring
/// `crate::extract::tests::extract_empty_samples_errors`'s identical
/// rationale.
#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn fluid_audio_source_empty_samples_errors_like_extractor() {
  let seg_a = load_seg_model();
  let embed_a = load_embed_model();
  let want = Extractor::new().extract(&seg_a, &embed_a, &[]);

  let seg_b = load_seg_model();
  let embed_b = load_embed_model();
  let got = FluidAudioSource::new(seg_b, embed_b).extract(&[]);

  assert_eq!(got, want);
  assert_eq!(got, Err(ExtractError::EmptySamples));
}
