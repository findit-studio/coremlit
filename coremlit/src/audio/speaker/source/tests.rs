use super::*;
use crate::ComputeUnits;

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
// (`any_source_load_dispatches_fluid_audio_and_argmax`, model-gated).

/// Loading `Source::Argmax` from a directory that holds only the FluidAudio
/// artifacts must FAIL, not silently fall back to `FluidAudioSource` — the
/// central honesty property of the dispatcher. (Hermetic: the load fails on
/// a nonexistent path, no models needed.)
#[test]
fn any_source_argmax_does_not_fall_back_to_fluid_audio() {
  let nowhere = std::path::Path::new("/nonexistent-speakerkit-models");
  let got = AnySource::load(nowhere, Options::new().with_source(Source::Argmax));
  assert!(
    matches!(got, Err(crate::audio::speaker::error::ModelError::Load(_))),
    "a missing argmax model must surface as a load error, never a \
     FluidAudio source; got {got:?}"
  );
  // And the FluidAudio arm fails independently on the same missing path.
  let got = AnySource::load(nowhere, Options::new().with_source(Source::FluidAudio));
  assert!(matches!(
    got,
    Err(crate::audio::speaker::error::ModelError::Load(_))
  ));
}

/// Finding 3, hermetic: the shipping FluidAudio selection is a pure function of
/// the models root, pinned to the fp32 `wespeaker.mlmodelc` (issue #15 — the
/// int8 `wespeaker_v2.mlmodelc` is retired from shipping; its palettization
/// collapses 8-speaker audio, see `tests/speaker/model_io.rs`'s DECISION).
/// This is the same resolver [`AnySource::load`] uses, so the pin sits on
/// production's own selection. Repointing [`FluidAudioArtifacts::resolve`]
/// back at `wespeaker_v2.mlmodelc` fails this immediately — no models needed,
/// because the resolver does no I/O.
#[test]
fn fluid_audio_artifacts_resolve_to_the_fp32_shipping_embedder() {
  let artifacts = FluidAudioArtifacts::resolve("some/models/root");
  assert!(
    artifacts.embedder().ends_with("wespeaker.mlmodelc"),
    "the shipping FluidAudio embedder must be the fp32 wespeaker.mlmodelc, got {}",
    artifacts.embedder().display()
  );
  assert!(
    artifacts
      .segmenter()
      .ends_with("pyannote_segmentation.mlmodelc"),
    "the FluidAudio segmenter must be pyannote_segmentation.mlmodelc, got {}",
    artifacts.segmenter().display()
  );
  // Rooted under the given directory, both files (a full-path pin, so a change
  // to the join logic — not just the filename — also fails).
  assert_eq!(
    artifacts.embedder(),
    std::path::Path::new("some/models/root/wespeaker.mlmodelc")
  );
  assert_eq!(
    artifacts.segmenter(),
    std::path::Path::new("some/models/root/pyannote_segmentation.mlmodelc")
  );
}

// =====================================================================
// Hermetic: FluidAudioArtifactConfig (the declarative artifact layer)
// =====================================================================

/// The wrap is transparent: at the empty config, `resolve_with` produces the
/// SAME fixed-name selection `resolve` has always produced.
///
/// The load-bearing assertions are the LITERAL expected paths — comparing
/// `resolve_with(root, &new())` against `resolve(root)` alone would be vacuous,
/// since `resolve` now delegates to exactly that call. The literals are what
/// fails if the join logic, either filename, or the `None` fallback moves; the
/// equality line only records that the delegation is still in place.
///
/// Several root shapes, because "byte-identical" is a claim about the join, not
/// just about one happy path: relative, absolute, empty, and trailing-separator
/// roots all have distinct `Path::join` behaviour.
#[test]
fn resolve_with_default_config_matches_the_pinned_fixed_names() {
  const CASES: &[(&str, &str, &str)] = &[
    (
      "some/models/root",
      "some/models/root/pyannote_segmentation.mlmodelc",
      "some/models/root/wespeaker.mlmodelc",
    ),
    (
      "/abs/Models/speakerkit",
      "/abs/Models/speakerkit/pyannote_segmentation.mlmodelc",
      "/abs/Models/speakerkit/wespeaker.mlmodelc",
    ),
    (
      "trailing/",
      "trailing/pyannote_segmentation.mlmodelc",
      "trailing/wespeaker.mlmodelc",
    ),
    ("", "pyannote_segmentation.mlmodelc", "wespeaker.mlmodelc"),
  ];

  let config = FluidAudioArtifactConfig::new();
  for (root, want_seg, want_embed) in CASES {
    let got = FluidAudioArtifacts::resolve_with(root, &config);
    assert_eq!(
      got.segmenter(),
      std::path::Path::new(want_seg),
      "default-config segmenter under root {root:?}"
    );
    assert_eq!(
      got.embedder(),
      std::path::Path::new(want_embed),
      "default-config embedder under root {root:?}"
    );
    assert_eq!(
      got,
      FluidAudioArtifacts::resolve(root),
      "resolve must stay a delegation to resolve_with at the empty config"
    );
  }
  // The empty config is also what `Default` yields, so a caller who
  // `Default::default()`s one gets the pinned selection too.
  assert_eq!(config, FluidAudioArtifactConfig::default());
}

/// An explicit path is used VERBATIM: not joined onto `models_root`, not
/// normalized, and free to point outside the root entirely.
#[test]
fn resolve_with_uses_an_explicit_path_verbatim() {
  let config = FluidAudioArtifactConfig::new()
    .with_segmenter("/elsewhere/my_seg.mlmodelc")
    .with_embedder("relative/my_embed.mlmodelc");

  let got = FluidAudioArtifacts::resolve_with("some/models/root", &config);
  assert_eq!(
    got.segmenter(),
    std::path::Path::new("/elsewhere/my_seg.mlmodelc")
  );
  assert_eq!(
    got.embedder(),
    std::path::Path::new("relative/my_embed.mlmodelc")
  );
  // Verbatim, not rooted: a `root.join(override)` implementation would put both
  // under the root and pass the equality above only by accident, so assert the
  // root is absent from each.
  assert!(!got.segmenter().starts_with("some/models/root"));
  assert!(!got.embedder().starts_with("some/models/root"));
}

/// The two fields fall back INDEPENDENTLY — overriding one artifact leaves the
/// other on the convention. An all-or-nothing implementation (any override
/// replacing both, or none) fails both halves.
#[test]
fn resolve_with_falls_back_per_field_independently() {
  let seg_only =
    FluidAudioArtifacts::resolve_with("root", &FluidAudioArtifactConfig::new().with_segmenter("A"));
  assert_eq!(seg_only.segmenter(), std::path::Path::new("A"));
  assert_eq!(
    seg_only.embedder(),
    std::path::Path::new("root/wespeaker.mlmodelc"),
    "an omitted embedder must keep the convention when the segmenter is overridden"
  );

  let embed_only =
    FluidAudioArtifacts::resolve_with("root", &FluidAudioArtifactConfig::new().with_embedder("B"));
  assert_eq!(embed_only.embedder(), std::path::Path::new("B"));
  assert_eq!(
    embed_only.segmenter(),
    std::path::Path::new("root/pyannote_segmentation.mlmodelc"),
    "an omitted segmenter must keep the convention when the embedder is overridden"
  );
}

/// Per-model compute placement REACHES the options each model is loaded with,
/// and the config outranks the caller's [`Options`] when it names one.
///
/// `resolve_model_options` is the only place `load_with` obtains those two
/// values, and it returns them as the two DISTINCT option types, so the load
/// site can neither recompute nor transpose them.
#[test]
fn config_compute_reaches_the_model_load_options_and_outranks_options() {
  use crate::audio::speaker::{embed::EmbedModelOptions, segment::SegmentModelOptions};

  // Options say CpuAndGpu / CpuAndNeuralEngine; the config overrides both with
  // values distinct from those AND from the crate default (All), so no arm can
  // pass by coincidence.
  let options = Options::new().with_compute(
    crate::audio::speaker::extract::ComputeOptions::new()
      .with_segmenter(ComputeUnits::CpuAndGpu)
      .with_embedder(ComputeUnits::CpuAndNeuralEngine),
  );
  let config = FluidAudioArtifactConfig::new()
    .with_segmenter_compute(ComputeUnits::CpuOnly)
    .with_embedder_compute(ComputeUnits::All);

  let (seg, embed) = resolve_model_options(options, &config);
  assert_eq!(
    seg,
    SegmentModelOptions::new().with_compute(ComputeUnits::CpuOnly)
  );
  assert_eq!(
    embed,
    EmbedModelOptions::new().with_compute(ComputeUnits::All)
  );

  // Independently per model: overriding only the embedder must leave the
  // segmenter on the Options value.
  let (seg, embed) = resolve_model_options(
    options,
    &FluidAudioArtifactConfig::new().with_embedder_compute(ComputeUnits::CpuOnly),
  );
  assert_eq!(seg.compute(), ComputeUnits::CpuAndGpu);
  assert_eq!(embed.compute(), ComputeUnits::CpuOnly);
}

/// The other half of the precedence contract: an ABSENT config placement defers
/// to the caller's [`Options`], and an absent one there in turn leaves the
/// crate default. Three levels, each with a value the next cannot produce.
#[test]
fn absent_config_compute_defers_to_options_then_to_the_crate_default() {
  let empty = FluidAudioArtifactConfig::new();

  // Level 2: Options carries non-default placements; the empty config must not
  // overwrite them with the crate default.
  let options = Options::new().with_compute(
    crate::audio::speaker::extract::ComputeOptions::new()
      .with_segmenter(ComputeUnits::CpuOnly)
      .with_embedder(ComputeUnits::CpuAndGpu),
  );
  let (seg, embed) = resolve_model_options(options, &empty);
  assert_eq!(seg.compute(), ComputeUnits::CpuOnly);
  assert_eq!(embed.compute(), ComputeUnits::CpuAndGpu);

  // Level 3: both absent — the crate defaults, unchanged from today's
  // `AnySource::load`.
  let (seg, embed) = resolve_model_options(Options::new(), &empty);
  assert_eq!(
    seg.compute(),
    crate::audio::speaker::segment::DEFAULT_SEGMENT_COMPUTE
  );
  assert_eq!(
    embed.compute(),
    crate::audio::speaker::embed::DEFAULT_EMBED_COMPUTE
  );
}

/// End-to-end wiring, hermetic: the path the config names is the path the model
/// loader actually opens. [`crate::LoadError::NotFound`] carries the path it
/// checked, so a missing artifact reports exactly which file was selected.
///
/// The two overrides are DIFFERENT, and `load_with` loads the segmenter first,
/// so this also rules out a transposed call site: pairing the segmenter's
/// options with the embedder's path would report the embedder path here.
#[test]
fn load_with_opens_the_configured_segmenter_path() {
  let config = FluidAudioArtifactConfig::new()
    .with_segmenter("/nonexistent-byo/my_seg.mlmodelc")
    .with_embedder("/nonexistent-byo/my_embed.mlmodelc");

  let got = FluidAudioSource::load_with("/nonexistent-speakerkit-models", Options::new(), &config);
  match got {
    Err(crate::audio::speaker::error::ModelError::Load(crate::LoadError::NotFound(path))) => {
      assert_eq!(
        path,
        std::path::PathBuf::from("/nonexistent-byo/my_seg.mlmodelc"),
        "load_with must open the CONFIGURED segmenter, not the conventional \
         name and not the configured embedder"
      );
    }
    other => panic!("expected a NotFound naming the configured segmenter, got {other:?}"),
  }

  // Control: with the empty config the same call reports the conventional name
  // under the root, so the assertion above is about the override and not about
  // `load_with` always echoing its first argument.
  let got = FluidAudioSource::load("/nonexistent-speakerkit-models", Options::new());
  match got {
    Err(crate::audio::speaker::error::ModelError::Load(crate::LoadError::NotFound(path))) => {
      assert_eq!(
        path,
        std::path::PathBuf::from("/nonexistent-speakerkit-models/pyannote_segmentation.mlmodelc")
      );
    }
    other => panic!("expected a NotFound naming the conventional segmenter, got {other:?}"),
  }
}

/// [`FluidAudioSource::load_with`] does not consult `options`'s
/// [`Options::source`] when choosing WHAT TO LOAD: it always builds the
/// FluidAudio pair, exactly as [`Extractor::extract`] always runs the FluidAudio
/// orchestration whatever that field says.
///
/// "Not consulted" is narrower than "changes nothing", and the difference
/// matters: the selector is preserved verbatim in the stored [`Options`] and
/// stays observable afterwards through [`FluidAudioSource::options_ref`]. It is
/// the LOAD that ignores it.
///
/// What this test reaches is the two pure inputs to that load — the resolved
/// per-model options, and the first path opened. It CANNOT see anything past the
/// first model open, because a nonexistent segmenter returns before the embedder
/// is touched or the source is built; a branch on the selector after that point
/// is invisible here. That case belongs to
/// `load_with_constructs_the_same_pair_whatever_the_selector_says`, which loads
/// both models for real.
#[test]
fn load_with_does_not_consult_the_source_selector_when_choosing_what_to_load() {
  const ROOT: &str = "/nonexistent-speakerkit-models";
  let config = FluidAudioArtifactConfig::new();
  let base = Options::new().with_compute(
    crate::audio::speaker::extract::ComputeOptions::new()
      .with_segmenter(ComputeUnits::CpuOnly)
      .with_embedder(ComputeUnits::CpuAndGpu),
  );

  // The per-model load options are invariant across every `Source`. This is the
  // branch a path-only test cannot see: altering EITHER model's options on the
  // selector lands here, before any file is opened.
  let want = resolve_model_options(base, &config);
  for source in [Source::FluidAudio, Source::Argmax] {
    assert_eq!(
      resolve_model_options(base.with_source(source), &config),
      want,
      "resolved per-model options must not vary with Options::source ({source:?})"
    );
  }

  // And the first path actually opened is the FluidAudio segmenter even when the
  // selector names the other source — not argmax's `speaker_segmenter/` tree,
  // and not a refusal.
  let got = FluidAudioSource::load_with(ROOT, base.with_source(Source::Argmax), &config);
  match &got {
    Err(crate::audio::speaker::error::ModelError::Load(crate::LoadError::NotFound(path))) => {
      assert_eq!(
        *path,
        std::path::PathBuf::from("/nonexistent-speakerkit-models/pyannote_segmentation.mlmodelc"),
        "load_with must open the FluidAudio segmenter whatever Options::source names"
      );
    }
    other => panic!("expected a NotFound naming the FluidAudio segmenter, got {other:?}"),
  }
}

// ---------------------------------------------------------------------
// Hermetic: the serde wire contract (feature `serde`)
// ---------------------------------------------------------------------

/// THE gate that justifies `deny_unknown_fields`: a misspelled key is REJECTED
/// rather than silently ignored — which would load the pinned artifact while
/// the caller believes their own is running (the issue-#15 failure class).
///
/// Each rejection is paired with the spelling that must still be accepted, so
/// the test cannot pass because the whole struct stopped deserializing.
#[cfg(feature = "serde")]
#[test]
fn config_rejects_an_unknown_field() {
  // Positive control first: the correct key parses and lands in the field.
  let ok: FluidAudioArtifactConfig =
    serde_json::from_str(r#"{"embedder":"models/mine.mlmodelc"}"#).expect("the correct key parses");
  assert_eq!(
    ok.embedder(),
    Some(std::path::Path::new("models/mine.mlmodelc"))
  );

  // One character off, alone.
  let err =
    serde_json::from_str::<FluidAudioArtifactConfig>(r#"{"embeder":"models/mine.mlmodelc"}"#)
      .expect_err("a misspelled key must be rejected, never silently dropped");
  assert!(
    err.to_string().contains("embeder"),
    "the error must name the offending key; got {err}"
  );

  // And alongside valid keys, where a permissive impl is likeliest to let it
  // through: everything else here is well-formed.
  let err = serde_json::from_str::<FluidAudioArtifactConfig>(
    r#"{"segmenter":"a.mlmodelc","embeder":"b.mlmodelc","embedder_compute":"all"}"#,
  )
  .expect_err("an unknown key must be rejected even among valid ones");
  assert!(err.to_string().contains("embeder"), "got {err}");

  // A plausible near-miss on a compute key, too — the same silent-fallback risk.
  let err = serde_json::from_str::<FluidAudioArtifactConfig>(r#"{"embed_compute":"all"}"#)
    .expect_err("an unknown compute key must be rejected");
  assert!(err.to_string().contains("embed_compute"), "got {err}");
}

/// An omitted key means ABSENT, not "the default value": every field may be
/// left out, and the result is the empty config, which resolves to the
/// conventional selection.
#[cfg(feature = "serde")]
#[test]
fn config_omitted_fields_deserialize_as_absent() {
  let empty: FluidAudioArtifactConfig =
    serde_json::from_str("{}").expect("every field is optional");
  assert_eq!(empty, FluidAudioArtifactConfig::new());
  assert_eq!(empty.segmenter(), None);
  assert_eq!(empty.embedder(), None);
  assert_eq!(empty.segmenter_compute(), None);
  assert_eq!(empty.embedder_compute(), None);
  assert_eq!(
    FluidAudioArtifacts::resolve_with("root", &empty),
    FluidAudioArtifacts::resolve("root")
  );

  // Partial: one path named, everything else absent — the omitted compute keys
  // must not materialize as `Some(All)` (which would silently outrank a
  // caller's `Options`).
  let partial: FluidAudioArtifactConfig =
    serde_json::from_str(r#"{"segmenter":"a.mlmodelc"}"#).expect("a partial config parses");
  assert_eq!(
    partial.segmenter(),
    Some(std::path::Path::new("a.mlmodelc"))
  );
  assert_eq!(partial.embedder(), None);
  assert_eq!(partial.segmenter_compute(), None);
  assert_eq!(partial.embedder_compute(), None);
}

/// The TOML document shown in [`FluidAudioArtifactConfig`]'s "Wire format"
/// section, parsed verbatim.
///
/// That example previously carried a `[speaker]` header and could not parse into
/// this type at all: the header makes `speaker` a key of the ENCLOSING document,
/// which `deny_unknown_fields` rejects. A documented example that does not parse
/// is exactly the silently-wrong-configuration class this type exists to
/// prevent, so the snippet is executed here rather than trusted.
#[cfg(feature = "serde")]
#[test]
fn config_parses_the_documented_toml() {
  let config: FluidAudioArtifactConfig = toml::from_str(
    r#"
segmenter         = "models/my_seg.mlmodelc"
embedder          = "models/my_embed.mlmodelc"
segmenter_compute = "cpu_only"
embedder_compute  = "all"
"#,
  )
  .expect("the documented top-level TOML document must deserialize into this type");

  assert_eq!(
    config.segmenter(),
    Some(std::path::Path::new("models/my_seg.mlmodelc"))
  );
  assert_eq!(
    config.embedder(),
    Some(std::path::Path::new("models/my_embed.mlmodelc"))
  );
  assert_eq!(config.segmenter_compute(), Some(ComputeUnits::CpuOnly));
  assert_eq!(config.embedder_compute(), Some(ComputeUnits::All));

  // And it drives the resolver, which is the point of parsing it at all.
  let artifacts = FluidAudioArtifacts::resolve_with("Models/speakerkit", &config);
  assert_eq!(
    artifacts.embedder(),
    std::path::Path::new("models/my_embed.mlmodelc")
  );
}

/// The other half of that documented contract: a `[speaker]`-headed document is
/// REJECTED by this type, and the documented remedy — an outer struct owning a
/// `speaker` field — accepts the SAME document.
///
/// Both directions on one input are what make the explanation checkable: the
/// rejection alone could be any parse error, and the wrapper alone would not
/// show that the bare type refuses it.
#[cfg(feature = "serde")]
#[test]
fn config_rejects_a_speaker_headed_document_but_a_wrapper_accepts_it() {
  const DOC: &str = r#"
[speaker]
embedder = "models/my_embed.mlmodelc"
"#;

  let err = toml::from_str::<FluidAudioArtifactConfig>(DOC)
    .expect_err("a `[speaker]` header is a key of the enclosing document, not this struct");
  assert!(
    err.to_string().contains("speaker"),
    "the error must name the offending key; got {err}"
  );

  #[derive(serde::Deserialize)]
  struct AppConfig {
    speaker: FluidAudioArtifactConfig,
  }
  let app: AppConfig =
    toml::from_str(DOC).expect("the documented wrapper must accept the same document");
  assert_eq!(
    app.speaker.embedder(),
    Some(std::path::Path::new("models/my_embed.mlmodelc"))
  );
  // Nesting changes the document shape, never the omission semantics.
  assert_eq!(app.speaker.segmenter(), None);
  assert_eq!(app.speaker.embedder_compute(), None);
}

/// The full wire form: paths plus the snake_case `ComputeUnits` spelling shared
/// with [`crate::audio::speaker::extract::ComputeOptions`], round-tripping
/// through the option bridge. A `None` field must be OMITTED on the way out —
/// TOML, the format this config is shaped for, cannot encode a null.
#[cfg(feature = "serde")]
#[test]
fn config_round_trips_paths_and_compute_names() {
  let full: FluidAudioArtifactConfig = serde_json::from_str(
    r#"{
      "segmenter": "models/my_seg.mlmodelc",
      "embedder": "models/my_embed.mlmodelc",
      "segmenter_compute": "cpu_only",
      "embedder_compute": "cpu_and_neural_engine"
    }"#,
  )
  .expect("the documented wire form parses");
  assert_eq!(
    full,
    FluidAudioArtifactConfig::new()
      .with_segmenter("models/my_seg.mlmodelc")
      .with_embedder("models/my_embed.mlmodelc")
      .with_segmenter_compute(ComputeUnits::CpuOnly)
      .with_embedder_compute(ComputeUnits::CpuAndNeuralEngine)
  );
  let back: FluidAudioArtifactConfig =
    serde_json::from_str(&serde_json::to_string(&full).unwrap()).unwrap();
  assert_eq!(back, full);

  // An unparseable compute name is an error, not a fallback to the default.
  serde_json::from_str::<FluidAudioArtifactConfig>(r#"{"embedder_compute":"All"}"#)
    .expect_err("ComputeUnits names are the snake_case forms; `All` is not one");

  // Absent fields are omitted entirely on serialize (no nulls to encode).
  let json = serde_json::to_string(&FluidAudioArtifactConfig::new()).unwrap();
  assert_eq!(json, "{}", "a null-free encoding is required for TOML");
}

/// [`AnySource::load`] builds the source [`Options::source`] names — and
/// dispatches `extract` to it.
///
/// The name carries `argmax` because CI reads it. This is the only in-lib
/// speaker gate outside `source::argmax::tests` that needs
/// `ARGMAX_TEST_MODELS`, a tree no runner is allowed to fetch, and the
/// `speaker` shard excludes the whole set of eleven with one `--skip=argmax`
/// (.github/workflows/ci.yml, pinned in both directions by
/// `ci_speaker_lib_gates_skip_exactly_the_unstaged_argmax_tree`).
#[test]
#[ignore = "requires local argmax + speakerkit models (both env vars)"]
fn any_source_load_dispatches_fluid_audio_and_argmax() {
  let samples = load_ted_head();

  let fluid = AnySource::load(models_dir(), Options::new().with_source(Source::FluidAudio))
    .expect("load FluidAudio via the dispatcher");
  assert_eq!(fluid.source(), Source::FluidAudio);
  assert!(matches!(fluid, AnySource::FluidAudio(_)));
  let fluid_out = fluid.extract(&samples).expect("dispatched extract");

  let argmax_dir = std::env::var_os("ARGMAX_TEST_MODELS").map_or_else(
    || crate::tests::models_root().join("argmax-speakerkit"),
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
  // §4) — but nor are they required to DIFFER, so neither is asserted about
  // `segmentations`.
  //
  // Do NOT read the two sources as bit-identical segmenters. On this short,
  // clean, SINGLE-SPEAKER 2 s clip they do happen to produce identical
  // `segmentations`, but that is a property of THIS fixture, not of the two
  // models: on a 30 s two-speaker clip they disagree on 3 of 37 107
  // segmentation cells (0.008 %) and on ~65 % of embedding cells. The right
  // cross-source claim is decision-level near-agreement, never bit-identity
  // (Task 5 asserts it at that level).
  //
  // Only the embeddings must differ, because the two sources run entirely
  // different embedding networks (WeSpeaker vs argmax's `SpeakerEmbedder`) —
  // and both are non-trivial here, so this cannot pass vacuously on two zero
  // buffers.
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
// reason as crate::audio::speaker::extract::tests, crate::audio::speaker::embed::tests, and
// crate::audio::speaker::segment::tests: unit tests under `src/` cannot import the
// separate `tests/` integration-test crate.
// =====================================================================

fn models_dir() -> std::path::PathBuf {
  std::env::var_os("SPEAKERKIT_TEST_MODELS").map_or_else(
    || crate::tests::models_root().join("speakerkit"),
    std::path::PathBuf::from,
  )
}

fn load_seg_model() -> SegmentModel {
  // CpuOnly for determinism, matching crate::audio::speaker::extract::tests::load_seg_model
  // and every other model-gated loader in this crate.
  SegmentModel::from_file_with(
    models_dir().join("pyannote_segmentation.mlmodelc"),
    crate::audio::speaker::segment::SegmentModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load pyannote_segmentation.mlmodelc")
}

fn load_embed_model() -> EmbedModel {
  EmbedModel::from_file_with(
    models_dir().join("wespeaker_v2.mlmodelc"),
    crate::audio::speaker::embed::EmbedModelOptions::new().with_compute(ComputeUnits::CpuOnly),
  )
  .expect("load wespeaker_v2.mlmodelc")
}

/// The first 2 s (32_000 samples at 16 kHz) of the cross-crate `ted_60.wav`
/// fixture (see `crate::audio::speaker::extract::tests::load_ted_60` for the full-clip
/// loader) — long enough to be a real, non-degenerate segmentation chunk,
/// short enough (`<= SEG_CHUNK_SAMPLES`) that `crate::audio::speaker::window::chunk_starts`
/// always yields exactly one chunk, keeping these equivalence tests fast.
fn load_ted_head() -> Vec<f32> {
  let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("tests/whisper/fixtures/audio/ted_60.wav");
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
    crate::audio::speaker::window::WindowOptions::new()
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
/// `crate::audio::speaker::extract::tests::extract_empty_samples_errors`'s identical
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

/// The embedder half of `load_with_opens_the_configured_segmenter_path`, which
/// a hermetic test cannot reach: the segmenter has to load successfully before
/// the embedder path is ever opened. Only the embedder is redirected here, so
/// the reported [`crate::LoadError::NotFound`] path proves `load_with` opens the
/// CONFIGURED embedder rather than the conventional name.
#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn load_with_opens_the_configured_embedder_path() {
  let config = FluidAudioArtifactConfig::new()
    .with_embedder("/nonexistent-byo/my_embed.mlmodelc")
    .with_segmenter_compute(ComputeUnits::CpuOnly);

  let got = FluidAudioSource::load_with(models_dir(), Options::new(), &config);
  match got {
    Err(crate::audio::speaker::error::ModelError::Load(crate::LoadError::NotFound(path))) => {
      assert_eq!(
        path,
        std::path::PathBuf::from("/nonexistent-byo/my_embed.mlmodelc")
      );
    }
    other => panic!("expected a NotFound naming the configured embedder, got {other:?}"),
  }
}

/// The half `load_with_does_not_consult_the_source_selector_when_choosing_what_to_load`
/// cannot reach: a SUCCESSFUL construction.
///
/// A regression that branches on `options.source()` AFTER the segmenter opens —
/// refusing, rerouting the embedder, or changing its placement — is invisible to
/// a nonexistent-path test, which returns from the first open. Both models load
/// here, so the selector's effect on the whole construction is observable: the
/// two sources must behave identically on real audio, while the selector itself
/// is PRESERVED in the stored [`Options`] rather than erased.
#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn load_with_constructs_the_same_pair_whatever_the_selector_says() {
  let base = Options::new().with_compute(
    crate::audio::speaker::extract::ComputeOptions::new()
      .with_segmenter(ComputeUnits::CpuOnly)
      .with_embedder(ComputeUnits::CpuOnly),
  );
  let config = FluidAudioArtifactConfig::new();

  let fluid =
    FluidAudioSource::load_with(models_dir(), base, &config).expect("load at the default selector");
  let argmax_selected =
    FluidAudioSource::load_with(models_dir(), base.with_source(Source::Argmax), &config)
      .expect("an Argmax selector must not refuse the load, nor reroute it to a missing artifact");

  // Preserved, not erased: the stored `Options` is the caller's, verbatim.
  assert_eq!(fluid.options_ref().source(), Source::FluidAudio);
  assert_eq!(argmax_selected.options_ref().source(), Source::Argmax);

  // ...and it changed nothing about WHICH artifacts were loaded or how. A
  // rerouted embedder path or a different placement diverges here.
  let samples = load_ted_head();
  let want = fluid.extract(&samples).expect("default-selector extract");
  let got = argmax_selected
    .extract(&samples)
    .expect("argmax-selector extract");
  assert_eq!(
    got, want,
    "the selector must not affect which models were loaded, nor how"
  );
  assert!(
    want.raw_embeddings().iter().any(|&v| v != 0.0),
    "the comparison above must not be two zero buffers"
  );
}

/// A configured artifact does not merely reach the loader — it is the model that
/// RUNS. Swapping in the retired int8 `wespeaker_v2.mlmodelc` (contract-identical
/// to the shipping fp32 artifact, which is exactly why it passes every
/// shape/dtype gate) changes the embeddings while leaving segmentation
/// untouched.
///
/// This is the documented hazard in executable form: the swap is silent at every
/// level this crate checks automatically.
#[test]
#[ignore = "requires local speakerkit models (SPEAKERKIT_TEST_MODELS)"]
fn load_with_a_custom_embedder_runs_that_model() {
  let samples = load_ted_head();
  // CpuOnly on both models for determinism, matching every other model-gated
  // loader in this crate.
  let options = Options::new().with_compute(
    crate::audio::speaker::extract::ComputeOptions::new()
      .with_segmenter(ComputeUnits::CpuOnly)
      .with_embedder(ComputeUnits::CpuOnly),
  );

  let shipping =
    FluidAudioSource::load(models_dir(), options).expect("load the shipping fp32 artifacts");
  let custom = FluidAudioSource::load_with(
    models_dir(),
    options,
    &FluidAudioArtifactConfig::new().with_embedder(models_dir().join("wespeaker_v2.mlmodelc")),
  )
  .expect("load the retired int8 embedder through the config");

  // `options` threads into the built source, exactly as `with_options` does.
  assert_eq!(custom.options_ref(), &options);

  let want = shipping.extract(&samples).expect("shipping extract");
  let got = custom.extract(&samples).expect("custom-embedder extract");

  assert_eq!(
    got.segmentations(),
    want.segmentations(),
    "only the embedder was substituted; segmentation must be untouched"
  );
  assert!(
    want.raw_embeddings().iter().any(|&v| v != 0.0),
    "the comparison below must not be two zero buffers"
  );
  assert_ne!(
    got.raw_embeddings(),
    want.raw_embeddings(),
    "the configured artifact must be the model that actually ran"
  );
}
