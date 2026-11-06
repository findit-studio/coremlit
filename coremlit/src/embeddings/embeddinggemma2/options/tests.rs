use super::*;
use crate::{Specialization, embeddings::aggregate::EmaRenormalizedOptions};
use windit::{TailPolicy, plan::WindowOptions};

use super::super::text::TextFunction;

fn digest_label() -> ArtifactLabel {
  ArtifactLabel::from_digests([("a/b.bin", [0x11; 32]), ("Z.json", [0x00; 32])])
}

/// A text configuration with every option moved off its default, the bounds
/// and the non-vector options included.
fn moved_text() -> TextOptions {
  TextOptions::new()
    .with_dir("/models/eg2")
    .with_coreml(
      LoadOptions::new(ComputeUnits::All)
        .with_low_precision_gpu_accumulation(true)
        .with_specialization(Specialization::FastPrediction),
    )
    .with_prewarm(TextFunctions::empty().with(TextFunction::Embed32))
    .with_batching(Batching::Single)
    .with_split_special_tokens(true)
    .with_long_text(
      LongTextOptions::from(
        WindowOptions::new(256)
          .with_overlap(32)
          .with_tail(TailPolicy::DropBelowMin(3))
          .with_max_windows(8),
      )
      .with_max_input_bytes(4096),
    )
    .with_fold(AggregatePolicyKind::EmaRenormalized(
      EmaRenormalizedOptions::new(0.25),
    ))
    .with_expect_artifacts(ArtifactLabel::FLUIDINFERENCE)
}

fn moved_audio() -> AudioOptions {
  AudioOptions::new()
    .with_dir("/models/eg2-audio")
    .with_precision(AudioPrecision::Fp32)
    .with_coreml(LoadOptions::new(ComputeUnits::CpuOnly))
    .with_batching(Batching::Packed)
    .with_hop_samples(NonZeroU32::new(80_000).unwrap())
    .with_min_tail_samples(0)
    .with_max_windows(NonZeroU32::new(360).unwrap())
    .with_expect_artifacts(digest_label())
}

fn moved_image() -> ImageOptions {
  ImageOptions::new()
    .with_dir("/models/eg2-vision")
    .with_coreml(
      LoadOptions::new(ComputeUnits::CpuAndGpu).with_low_precision_gpu_accumulation(true),
    )
    .with_batching(Batching::Packed)
    .with_budgets(VisionBudgets::all())
    .with_default_budget(VisionBudget::Tokens280)
    .with_expect_artifacts(ArtifactLabel::FLUIDINFERENCE)
}

/// K2: the derivation terms of the defaults over the pinned upstream
/// artifacts, part by part.
#[test]
fn the_default_derivation_terms_are_pinned() {
  let label = ArtifactLabel::FLUIDINFERENCE;
  let text = TextOptions::new();
  let audio = AudioOptions::new();
  let image = ImageOptions::new();
  assert_eq!(
    DerivationTerms::text(label, &text).to_string(),
    "embeddinggemma2/text(artifacts=fluidinference@3f978dd,coreml=(compute=cpu_and_neural_engine,low_precision_gpu_accumulation=false,specialization=default),batching=packed,split_special_tokens=false,window=(window=512,hop=512,tail=keep_with_coverage),fold=coverage_weighted_mean)"
  );
  assert_eq!(
    DerivationTerms::audio(label, &audio).to_string(),
    "embeddinggemma2/audio(artifacts=fluidinference@3f978dd,precision=fp16,coreml=(compute=cpu_and_gpu,low_precision_gpu_accumulation=false,specialization=default),batching=single,hop_samples=160000,min_tail_samples=8000)"
  );
  assert_eq!(
    DerivationTerms::vision(label, &image).to_string(),
    "embeddinggemma2/vision(artifacts=fluidinference@3f978dd,coreml=(compute=cpu_and_gpu,low_precision_gpu_accumulation=false,specialization=default),batching=single,default_budget=140,resize=hf_bicubic)"
  );
}

/// K2: every vector-changing option moves its term; the geometry is the
/// terms' own projection, so neither bound appears; the non-vector options
/// (`dir`, `prewarm`, `budgets`, `expect_artifacts`) never print; a digest
/// label prints in place of the revision.
#[test]
fn the_moved_derivation_terms_are_pinned_and_print_no_bound() {
  let label = digest_label();
  let text = moved_text();
  let spelled = DerivationTerms::text(label, &text).to_string();
  assert_eq!(
    spelled,
    "embeddinggemma2/text(artifacts=sha256:896a97a8cefa,coreml=(compute=all,low_precision_gpu_accumulation=true,specialization=fast_prediction),batching=single,split_special_tokens=true,window=(window=256,hop=224,tail=drop_below_min(3)),fold=ema_renormalized(alpha=0.25))"
  );
  for absent in [
    "max_windows",
    "max_input_bytes",
    "dir",
    "prewarm",
    "expect_artifacts",
    "/models",
  ] {
    assert!(!spelled.contains(absent), "`{absent}` in {spelled}");
  }
  let audio = moved_audio();
  assert_eq!(
    DerivationTerms::audio(label, &audio).to_string(),
    "embeddinggemma2/audio(artifacts=sha256:896a97a8cefa,precision=fp32,coreml=(compute=cpu_only,low_precision_gpu_accumulation=false,specialization=default),batching=packed,hop_samples=80000,min_tail_samples=0)"
  );
  let image = moved_image();
  assert_eq!(
    DerivationTerms::vision(label, &image).to_string(),
    "embeddinggemma2/vision(artifacts=sha256:896a97a8cefa,coreml=(compute=cpu_and_gpu,low_precision_gpu_accumulation=true,specialization=default),batching=packed,default_budget=280,resize=hf_bicubic)"
  );
}

/// L14: pinning the artifacts changes no derivation term, in any part.
#[test]
fn the_artifact_pin_enters_no_derivation_term() {
  let label = ArtifactLabel::FLUIDINFERENCE;
  let mut text = TextOptions::new();
  let before = DerivationTerms::text(label, &text).to_string();
  text.set_expect_artifacts(Some(digest_label()));
  assert_eq!(DerivationTerms::text(label, &text).to_string(), before);

  let mut audio = AudioOptions::new();
  let before = DerivationTerms::audio(label, &audio).to_string();
  audio.set_expect_artifacts(Some(digest_label()));
  assert_eq!(DerivationTerms::audio(label, &audio).to_string(), before);

  let mut image = ImageOptions::new();
  let before = DerivationTerms::vision(label, &image).to_string();
  image.set_expect_artifacts(Some(digest_label()));
  assert_eq!(DerivationTerms::vision(label, &image).to_string(), before);
}

/// K2: the options' canonical `Display`, defaults and moved.
#[test]
fn the_options_display_is_pinned() {
  assert_eq!(
    TextOptions::new().to_string(),
    "dir=none,coreml=(compute=cpu_and_neural_engine,low_precision_gpu_accumulation=false,specialization=default),prewarm=reachable,batching=packed,split_special_tokens=false,long_text=(window_options=(window=512,hop=512,tail=keep_with_coverage,max_windows=none),max_input_bytes=none),fold=coverage_weighted_mean,expect_artifacts=none"
  );
  assert_eq!(
    moved_text().to_string(),
    "dir=\"/models/eg2\",coreml=(compute=all,low_precision_gpu_accumulation=true,specialization=fast_prediction),prewarm=[embed_32],batching=single,split_special_tokens=true,long_text=(window_options=(window=256,hop=224,tail=drop_below_min(3),max_windows=8),max_input_bytes=4096),fold=ema_renormalized(alpha=0.25),expect_artifacts=fluidinference@3f978dd"
  );
  assert_eq!(
    AudioOptions::new().to_string(),
    "dir=none,precision=fp16,coreml=(compute=cpu_and_gpu,low_precision_gpu_accumulation=false,specialization=default),batching=single,hop_samples=160000,min_tail_samples=8000,max_windows=none,expect_artifacts=none"
  );
  assert_eq!(
    moved_audio().to_string(),
    "dir=\"/models/eg2-audio\",precision=fp32,coreml=(compute=cpu_only,low_precision_gpu_accumulation=false,specialization=default),batching=packed,hop_samples=80000,min_tail_samples=0,max_windows=360,expect_artifacts=sha256:896a97a8cefa"
  );
  assert_eq!(
    ImageOptions::new().to_string(),
    "dir=none,coreml=(compute=cpu_and_gpu,low_precision_gpu_accumulation=false,specialization=default),batching=single,budgets=[140],default_budget=140,resize=hf_bicubic,expect_artifacts=none"
  );
  assert_eq!(
    EmbedderOptions::new()
      .with_vision(ImageOptions::new())
      .to_string(),
    format!(
      "text=({}),audio=none,vision=({})",
      TextOptions::new(),
      ImageOptions::new()
    )
  );
}

/// K2: the batching spellings.
#[test]
fn the_batching_spellings_are_pinned() {
  assert_eq!(Batching::Packed.to_string(), "packed");
  assert_eq!(Batching::Single.as_str(), "single");
}

/// K4: a window over the model's 512 ids, and one windit refuses, are refused
/// by name before anything loads.
#[test]
fn an_over_budget_or_degenerate_window_is_refused_by_name() {
  let over = TextOptions::new().with_long_text(LongTextOptions::from(WindowOptions::new(513)));
  match over.validate() {
    Err(Error::WindowOverBudget(w)) => assert_eq!((w.window(), w.max()), (513, 512)),
    other => panic!("expected WindowOverBudget, got {other:?}"),
  }
  let zero = TextOptions::new().with_long_text(LongTextOptions::from(WindowOptions::new(0)));
  assert!(
    matches!(zero.validate(), Err(Error::Windowing(_))),
    "{:?}",
    zero.validate()
  );
  TextOptions::new()
    .validate()
    .expect("the defaults are valid");
  moved_text()
    .validate()
    .expect("a 256-id window with a 32-id overlap is valid");
  EmbedderOptions::new()
    .with_vision(ImageOptions::new())
    .validate()
    .expect("the defaults are valid");
}

/// K4: a default budget outside the loaded budgets is refused naming both.
#[test]
fn a_default_budget_outside_the_loaded_set_is_refused_by_name() {
  let image = ImageOptions::new().with_budgets(VisionBudgets::only(VisionBudget::Tokens70));
  match image.validate() {
    Err(Error::BudgetNotLoaded(b)) => {
      assert_eq!(
        (b.budget(), b.loaded()),
        (
          VisionBudget::Tokens140,
          VisionBudgets::only(VisionBudget::Tokens70)
        )
      );
    }
    other => panic!("expected BudgetNotLoaded, got {other:?}"),
  }
  match EmbedderOptions::new().with_vision(image).validate() {
    Err(e) => assert_eq!(
      e.to_string(),
      "vision budget 140 is not among the loaded budgets [70]; a budget is never loaded on the side"
    ),
    Ok(()) => panic!("the facade validates its vision part"),
  }
}

/// K4: the parts' Core ML defaults are `LoadOptions::new(units)`, which sends
/// nothing beyond the compute units.
#[test]
fn the_parts_coreml_defaults_send_nothing_new() {
  for (coreml, units) in [
    (
      TextOptions::new().coreml(),
      ComputeUnits::CpuAndNeuralEngine,
    ),
    (AudioOptions::new().coreml(), ComputeUnits::CpuAndGpu),
    (ImageOptions::new().coreml(), ComputeUnits::CpuAndGpu),
  ] {
    assert_eq!(coreml, LoadOptions::new(units));
    assert!(!coreml.low_precision_gpu_accumulation());
    assert_eq!(coreml.specialization(), Specialization::Default);
  }
}

#[cfg(feature = "serde")]
mod wire {
  use super::*;

  fn round_trip<T>(value: &T)
  where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + core::fmt::Debug,
  {
    let json = serde_json::to_string(value).unwrap();
    assert_eq!(
      &serde_json::from_str::<T>(&json).unwrap(),
      value,
      "JSON {json}"
    );
    let toml_doc = toml::to_string(value).unwrap();
    assert_eq!(
      &toml::from_str::<T>(&toml_doc).unwrap(),
      value,
      "TOML {toml_doc}"
    );
    let bytes = postcard::to_allocvec(value).unwrap();
    assert_eq!(
      &postcard::from_bytes::<T>(&bytes).unwrap(),
      value,
      "postcard"
    );
  }

  /// K2: every options type round-trips through JSON, TOML and postcard, at
  /// its defaults and with every option moved.
  #[test]
  fn the_options_round_trip_through_json_toml_and_postcard() {
    round_trip(&TextOptions::new());
    round_trip(&moved_text());
    round_trip(&AudioOptions::new());
    round_trip(&moved_audio());
    round_trip(&ImageOptions::new());
    round_trip(&moved_image());
    round_trip(&EmbedderOptions::new());
    round_trip(
      &EmbedderOptions::new()
        .with_text(moved_text())
        .with_audio(moved_audio())
        .with_vision(moved_image()),
    );
  }

  /// K2: an empty document is the defaults, and the document spells each
  /// vocabulary by its own word.
  #[test]
  fn an_empty_document_is_the_defaults_and_the_keys_are_pinned() {
    assert_eq!(
      serde_json::from_str::<TextOptions>("{}").unwrap(),
      TextOptions::new()
    );
    assert_eq!(
      serde_json::from_str::<AudioOptions>("{}").unwrap(),
      AudioOptions::new()
    );
    assert_eq!(
      serde_json::from_str::<ImageOptions>("{}").unwrap(),
      ImageOptions::new()
    );
    assert_eq!(
      serde_json::from_str::<EmbedderOptions>("{}").unwrap(),
      EmbedderOptions::new()
    );

    let doc = r#"
      [text]
      batching = "single"
      split_special_tokens = true
      prewarm = ["embed_128", "pack_256"]
      fold = { ema_renormalized = { alpha = 0.25 } }
      coreml = { compute = "cpu_only" }
      long_text = { window_options = { window = 256, hop = 192, tail = { kind = "keep_with_coverage" } } }

      [audio]
      precision = "fp32"
      hop_samples = 80000

      [vision]
      budgets = [140, 280]
      default_budget = 280
      expect_artifacts = "fluidinference@3f978dd"
    "#;
    let options: EmbedderOptions = toml::from_str(doc).unwrap();
    let text = options.text();
    assert_eq!(text.batching(), Batching::Single);
    assert!(text.split_special_tokens());
    assert_eq!(text.prewarm().unwrap().to_string(), "[embed_128,pack_256]");
    assert_eq!(text.fold().to_string(), "ema_renormalized(alpha=0.25)");
    assert_eq!(text.coreml(), LoadOptions::new(ComputeUnits::CpuOnly));
    assert_eq!(text.long_text().window_options().hop(), 192);
    let audio = options
      .audio()
      .expect("the [audio] table turns the lane on");
    assert_eq!(audio.precision(), AudioPrecision::Fp32);
    assert_eq!(audio.hop_samples().get(), 80_000);
    assert_eq!(audio.batching(), Batching::Single, "the face's own default");
    let vision = options
      .vision()
      .expect("the [vision] table turns the lane on");
    assert_eq!(vision.default_budget(), VisionBudget::Tokens280);
    assert_eq!(
      vision.expect_artifacts(),
      Some(ArtifactLabel::FLUIDINFERENCE)
    );
    assert_eq!(vision.batching(), Batching::Single);
  }

  /// K2: a vocabulary's document word is its name, so the two cannot drift.
  #[test]
  fn the_vocabulary_wire_words_are_their_names() {
    for batching in [Batching::Packed, Batching::Single] {
      assert_eq!(
        serde_json::to_string(&batching).unwrap(),
        format!("\"{}\"", batching.as_str())
      );
    }
    for precision in [AudioPrecision::Fp16, AudioPrecision::Fp32] {
      assert_eq!(
        serde_json::to_string(&precision).unwrap(),
        format!("\"{}\"", precision.as_str())
      );
    }
    let road = ResizeRoad::HfBicubic;
    assert_eq!(
      serde_json::to_string(&road).unwrap(),
      format!("\"{}\"", road.as_str())
    );
  }

  /// K4: an unknown key is refused naming it, at every level the kit owns.
  #[test]
  fn an_unknown_key_is_refused_at_every_level() {
    let cases: [(&str, &str); 10] = [
      (
        r#"{"text":{"split_special_token":true}}"#,
        "split_special_token",
      ),
      (
        r#"{"text":{"coreml":{"compute":"all","units":"all"}}}"#,
        "units",
      ),
      (
        r#"{"text":{"long_text":{"max_input_byte":4096}}}"#,
        "max_input_byte",
      ),
      (
        r#"{"text":{"long_text":{"window_options":{"window":512,"stride":1}}}}"#,
        "stride",
      ),
      (r#"{"audio":{"hop_sample":80000}}"#, "hop_sample"),
      (r#"{"vision":{"default_budgets":140}}"#, "default_budgets"),
      (r#"{"image":{}}"#, "image"),
      (r#"{"text":{"fold":"coverage"}}"#, "coverage"),
      (r#"{"vision":{"budgets":[140],"resize":"apple"}}"#, "apple"),
      // Codex round 1, finding 7: the shared EMA options too.
      (
        r#"{"text":{"fold":{"ema_renormalized":{"alpha":0.5,"alhpa":0.8}}}}"#,
        "alhpa",
      ),
    ];
    for (doc, key) in cases {
      let refused = serde_json::from_str::<EmbedderOptions>(doc)
        .unwrap_err()
        .to_string();
      assert!(refused.contains(key), "{doc}: {refused}");
    }
  }

  /// K4: a present `coreml` table follows `LoadOptions`' own form, so it must
  /// name its compute units; an absent one is the part's default.
  #[test]
  fn a_present_coreml_table_names_its_compute_units() {
    let refused =
      serde_json::from_str::<TextOptions>(r#"{"coreml":{"low_precision_gpu_accumulation":true}}"#)
        .unwrap_err()
        .to_string();
    assert!(refused.contains("compute"), "{refused}");
    assert_eq!(
      serde_json::from_str::<TextOptions>("{}").unwrap().coreml(),
      DEFAULT_TEXT_COREML
    );
  }

  /// L14: `expect_artifacts` is absent by default, present when set, a label
  /// in every format, and refused by name when it is not one.
  #[test]
  fn the_artifact_pin_round_trips_and_defaults_to_absent() {
    assert_eq!(
      serde_json::from_str::<TextOptions>("{}")
        .unwrap()
        .expect_artifacts(),
      None
    );
    assert_eq!(
      serde_json::from_str::<AudioOptions>("{}")
        .unwrap()
        .expect_artifacts(),
      None
    );
    assert_eq!(
      serde_json::from_str::<ImageOptions>("{}")
        .unwrap()
        .expect_artifacts(),
      None
    );
    for label in [ArtifactLabel::FLUIDINFERENCE, digest_label()] {
      round_trip(&TextOptions::new().with_expect_artifacts(label));
      round_trip(&AudioOptions::new().with_expect_artifacts(label));
      round_trip(&ImageOptions::new().with_expect_artifacts(label));
      let doc = format!("expect_artifacts = \"{label}\"\n");
      let text: TextOptions = toml::from_str(&doc).unwrap();
      assert_eq!(text.expect_artifacts(), Some(label));
    }
    let refused = toml::from_str::<AudioOptions>("expect_artifacts = \"fluidinference@latest\"\n")
      .unwrap_err()
      .to_string();
    assert!(refused.contains("is not an artifact label"), "{refused}");
  }
}

/// The coordinator's ruling (c) on G3: a media face packs in the TEXT model's
/// `pack_256`, so a packed face's batch independence is the packing table at
/// the TEXT part's compute units, whatever the face's own placement is.
/// `Single` answers true; a lane that is off answers as its default batching.
#[test]
fn a_packed_media_face_answers_at_the_text_placement() {
  use super::super::placement::PlacementTable;
  use crate::ComputeUnits;

  let table = PlacementTable::unmeasured(false).with(ComputeUnits::CpuAndNeuralEngine, true);
  let packed_on_gpu = |batching| {
    EmbedderOptions::new()
      .with_text(TextOptions::new().with_coreml(LoadOptions::new(ComputeUnits::CpuAndNeuralEngine)))
      .with_audio(
        AudioOptions::new()
          .with_batching(batching)
          .with_coreml(LoadOptions::new(ComputeUnits::CpuAndGpu)),
      )
      .with_vision(
        ImageOptions::new()
          .with_batching(batching)
          .with_coreml(LoadOptions::new(ComputeUnits::CpuAndGpu)),
      )
  };
  let packed = packed_on_gpu(Batching::Packed);
  assert!(
    packed.audio_batch_independent_in(&table),
    "the text placement's measurement answers for the audio face"
  );
  assert!(packed.image_batch_independent_in(&table));
  let text_on_gpu = packed
    .clone()
    .with_text(TextOptions::new().with_coreml(LoadOptions::new(ComputeUnits::CpuAndGpu)));
  assert!(!text_on_gpu.audio_batch_independent_in(&table));
  assert!(!text_on_gpu.image_batch_independent_in(&table));
  let single = packed_on_gpu(Batching::Single);
  assert!(single.audio_batch_independent_in(&table));
  assert!(single.image_batch_independent_in(&table));
  let lanes_off = EmbedderOptions::new();
  assert!(
    lanes_off.audio_batch_independent_in(&table) && lanes_off.image_batch_independent_in(&table),
    "an off lane answers as its default, single batching"
  );
}
