use super::*;

// ---------------------------------------------------------------------
// ModelState
// ---------------------------------------------------------------------

#[test]
fn state_machine_names_and_busy() {
  assert_eq!(ModelState::Loaded.as_str(), "loaded");
  assert!(ModelState::Prewarming.is_busy());
  assert!(!ModelState::Prewarmed.is_busy());
}

#[test]
fn state_every_variant_name_and_busy_set() {
  // Every as_str name, and the exact isBusy set from ModelState.swift:44-49
  // (Downloading|Prewarming|Loading|Unloading).
  assert_eq!(ModelState::Unloaded.as_str(), "unloaded");
  assert_eq!(ModelState::Downloading.as_str(), "downloading");
  assert_eq!(ModelState::Downloaded.as_str(), "downloaded");
  assert_eq!(ModelState::Prewarming.as_str(), "prewarming");
  assert_eq!(ModelState::Prewarmed.as_str(), "prewarmed");
  assert_eq!(ModelState::Loading.as_str(), "loading");
  assert_eq!(ModelState::Loaded.as_str(), "loaded");
  assert_eq!(ModelState::Unloading.as_str(), "unloading");

  for busy in [
    ModelState::Downloading,
    ModelState::Prewarming,
    ModelState::Loading,
    ModelState::Unloading,
  ] {
    assert!(busy.is_busy());
  }
  for idle in [
    ModelState::Unloaded,
    ModelState::Downloaded,
    ModelState::Prewarmed,
    ModelState::Loaded,
  ] {
    assert!(!idle.is_busy());
  }
}

#[test]
fn state_display_matches_as_str_not_swift_description() {
  // This crate's own as_str-drives-Display convention (Task/ChunkingStrategy
  // precedent), NOT Swift's separately UI-renamed description ("Specializing"/
  // "Specialized" for prewarming/prewarmed, ModelState.swift:36-37).
  assert_eq!(ModelState::Loaded.to_string(), "loaded");
  assert_eq!(ModelState::Prewarming.to_string(), "prewarming");
  assert_eq!(ModelState::Prewarmed.to_string(), "prewarmed");
}

#[test]
fn state_callback_type_is_constructible_and_send() {
  let cb: StateCallback = Box::new(|old, new| {
    assert_eq!(old, None);
    assert_eq!(new, ModelState::Loading);
  });
  cb(None, ModelState::Loading);

  fn assert_send<T: Send>(_: &T) {}
  assert_send(&cb);
}

// ---------------------------------------------------------------------
// ModelVariant / detect_variant
// ---------------------------------------------------------------------

#[test]
fn variant_detection_table() {
  assert_eq!(detect_variant(51865, 384), Some(ModelVariant::Tiny));
  assert_eq!(detect_variant(51864, 384), Some(ModelVariant::TinyEn));
  assert_eq!(detect_variant(51865, 512), Some(ModelVariant::Base));
  assert_eq!(detect_variant(51865, 768), Some(ModelVariant::Small));
  assert_eq!(detect_variant(51865, 1024), Some(ModelVariant::Medium));
  assert_eq!(detect_variant(51866, 1280), Some(ModelVariant::LargeV3));
  assert_eq!(detect_variant(51865, 1280), Some(ModelVariant::LargeV2));
  assert_eq!(detect_variant(999, 384), None);
  assert!(!ModelVariant::TinyEn.is_multilingual());
  assert_eq!(ModelVariant::LargeV2.as_str(), "large-v2");
}

#[test]
fn variant_detection_english_only_table() {
  assert_eq!(detect_variant(51864, 512), Some(ModelVariant::BaseEn));
  assert_eq!(detect_variant(51864, 768), Some(ModelVariant::SmallEn));
  assert_eq!(detect_variant(51864, 1024), Some(ModelVariant::MediumEn));
  // English-only vocab has no large variant in Swift's switch (no 1280 case);
  // an unrecognized encoder_dim within a recognized vocab is also None.
  assert_eq!(detect_variant(51864, 1280), None);
  assert_eq!(detect_variant(51865, 999), None);
}

#[test]
fn detect_variant_largev3_ignores_encoder_dim() {
  // Swift's largev3 branch (logitsDim == 51866) never consults encoderDim at
  // all (ModelUtilities.swift:164-166) -- verified against source, not brief.
  assert_eq!(detect_variant(51866, 384), Some(ModelVariant::LargeV3));
  assert_eq!(detect_variant(51866, 1), Some(ModelVariant::LargeV3));
}

#[test]
fn variant_as_str_matches_swift_description_exactly() {
  // Models.swift:62-83. Note the separator: `.en` uses a DOT, `-v2`/`-v3` use
  // a DASH -- these are not the same convention.
  assert_eq!(ModelVariant::Tiny.as_str(), "tiny");
  assert_eq!(ModelVariant::TinyEn.as_str(), "tiny.en");
  assert_eq!(ModelVariant::Base.as_str(), "base");
  assert_eq!(ModelVariant::BaseEn.as_str(), "base.en");
  assert_eq!(ModelVariant::Small.as_str(), "small");
  assert_eq!(ModelVariant::SmallEn.as_str(), "small.en");
  assert_eq!(ModelVariant::Medium.as_str(), "medium");
  assert_eq!(ModelVariant::MediumEn.as_str(), "medium.en");
  assert_eq!(ModelVariant::Large.as_str(), "large");
  assert_eq!(ModelVariant::LargeV2.as_str(), "large-v2");
  assert_eq!(ModelVariant::LargeV3.as_str(), "large-v3");
  assert_eq!(ModelVariant::LargeV3.to_string(), "large-v3");
}

#[test]
fn variant_multilingual_matches_swift() {
  // Models.swift:53-59.
  for v in [
    ModelVariant::Tiny,
    ModelVariant::Base,
    ModelVariant::Small,
    ModelVariant::Medium,
    ModelVariant::Large,
    ModelVariant::LargeV2,
    ModelVariant::LargeV3,
  ] {
    assert!(v.is_multilingual());
  }
  for v in [
    ModelVariant::TinyEn,
    ModelVariant::BaseEn,
    ModelVariant::SmallEn,
    ModelVariant::MediumEn,
  ] {
    assert!(!v.is_multilingual());
  }
}

#[test]
fn is_model_multilingual_checks_vocab_size() {
  // ModelUtilities.swift:124-126: `logitsDim != 51864`.
  assert!(is_model_multilingual(51865));
  assert!(is_model_multilingual(51866));
  assert!(!is_model_multilingual(51864));
}

// ---------------------------------------------------------------------
// detect_model_url
// ---------------------------------------------------------------------

#[test]
fn detect_model_url_prefers_compiled_over_package() {
  let dir = tempfile::tempdir().unwrap();
  std::fs::create_dir_all(dir.path().join("AudioEncoder.mlmodelc")).unwrap();
  std::fs::create_dir_all(
    dir
      .path()
      .join("AudioEncoder.mlpackage/Data/com.apple.CoreML"),
  )
  .unwrap();
  std::fs::write(
    dir
      .path()
      .join("AudioEncoder.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
    b"",
  )
  .unwrap();

  let found = detect_model_url(dir.path(), "AudioEncoder", false).unwrap();
  assert!(found.ends_with("AudioEncoder.mlmodelc"));
}

#[test]
fn detect_model_url_falls_back_to_package_when_compiled_missing() {
  let dir = tempfile::tempdir().unwrap();
  std::fs::create_dir_all(
    dir
      .path()
      .join("AudioEncoder.mlpackage/Data/com.apple.CoreML"),
  )
  .unwrap();
  std::fs::write(
    dir
      .path()
      .join("AudioEncoder.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
    b"",
  )
  .unwrap();

  let found = detect_model_url(dir.path(), "AudioEncoder", false).unwrap();
  assert!(found.ends_with("AudioEncoder.mlpackage/Data/com.apple.CoreML/model.mlmodel"));
}

#[test]
fn detect_model_url_errors_when_nothing_found() {
  let dir = tempfile::tempdir().unwrap();
  let err = detect_model_url(dir.path(), "Missing", false).unwrap_err();
  assert!(matches!(err, ModelError::NotFound { .. }));
}

#[test]
fn detect_model_url_recursive_finds_nested_compiled_bundle() {
  let dir = tempfile::tempdir().unwrap();
  let nested = dir.path().join("snapshots/abc123");
  std::fs::create_dir_all(nested.join("AudioEncoder.mlmodelc")).unwrap();

  let found = detect_model_url(dir.path(), "AudioEncoder", true).unwrap();
  assert!(found.ends_with("AudioEncoder.mlmodelc"));
  assert!(found.starts_with(&nested));
}

#[test]
fn detect_model_url_recursive_ignores_package_fallback() {
  // Swift's recursive overload never checks .mlpackage at all, at any depth
  // (ModelUtilities.swift:37-55) -- verified against source, not brief.
  let dir = tempfile::tempdir().unwrap();
  std::fs::create_dir_all(
    dir
      .path()
      .join("AudioEncoder.mlpackage/Data/com.apple.CoreML"),
  )
  .unwrap();
  std::fs::write(
    dir
      .path()
      .join("AudioEncoder.mlpackage/Data/com.apple.CoreML/model.mlmodel"),
    b"",
  )
  .unwrap();

  let err = detect_model_url(dir.path(), "AudioEncoder", true).unwrap_err();
  assert!(matches!(err, ModelError::NotFound { .. }));
}

// ---------------------------------------------------------------------
// glob_match
// ---------------------------------------------------------------------

#[test]
fn glob_match_truth_table() {
  assert!(glob_match("a*c", "abc"));
  assert!(glob_match("a?c", "abc"));
  assert!(!glob_match("a*c", "abx"));
  assert!(glob_match("*", "anything at all"));
}

#[test]
fn glob_match_edge_cases() {
  assert!(glob_match("*", ""));
  assert!(glob_match("abc", "abc"));
  assert!(!glob_match("abc", "abd"));
  assert!(!glob_match("a?c", "ac")); // ? requires exactly one char
  assert!(!glob_match("a?c", "abbc")); // ? is exactly one char, not one-or-more
  assert!(glob_match("**", "anything")); // consecutive stars behave as one
}

#[test]
fn glob_match_no_bracket_classes() {
  // Brackets are literal characters here, never a character class -- the
  // deliberate scope boundary this task's brief calls for (no call site in
  // argmax-oss-swift's `matching(glob:)` uses them).
  assert!(glob_match("a[bc]", "a[bc]"));
  assert!(!glob_match("a[bc]", "ab"));
}

#[test]
fn glob_match_real_call_site_pattern_crosses_path_separators() {
  // The exact download-pattern shape from ModelInfo::download_pattern / the
  // real Swift fixture (ModelDownloaderTests.swift:19-30): `*` must match
  // across `/`, matching `fnmatch(glob, $0, 0)` with no FNM_PATHNAME
  // (FoundationExtensions.swift:113-118).
  assert!(glob_match(
    "speaker_segmenter/pyannote-v3/W8A16/*",
    "speaker_segmenter/pyannote-v3/W8A16/SpeakerSegmenter.mlmodelc/coremldata.bin"
  ));
  assert!(!glob_match(
    "speaker_segmenter/pyannote-v3/W8A16/*",
    "speaker_embedder/pyannote-v3/W8A16/SpeakerEmbedder.mlmodelc/coremldata.bin"
  ));
}

// ---------------------------------------------------------------------
// ModelInfo
// ---------------------------------------------------------------------

#[test]
fn model_info_rejects_empty_name() {
  let err =
    ModelInfo::try_new("", None, None, coremlit::ComputeUnits::CpuAndNeuralEngine).unwrap_err();
  assert!(matches!(err, ModelError::EmptyName));
}

#[test]
fn model_info_download_pattern_full_and_minimal() {
  // ModelDownloaderTests.testModelInfoDownloadPattern (ModelDownloaderTests.swift:199-205):
  // missing fields become a literal "*", not an empty path segment.
  let full = ModelInfo::try_new(
    "speaker_segmenter",
    Some("pyannote-v3".to_string()),
    Some("W8A16".to_string()),
    coremlit::ComputeUnits::CpuAndNeuralEngine,
  )
  .unwrap();
  assert_eq!(
    full.download_pattern(),
    "speaker_segmenter/pyannote-v3/W8A16/*"
  );

  let minimal = ModelInfo::try_new(
    "speaker_segmenter",
    None,
    None,
    coremlit::ComputeUnits::CpuAndNeuralEngine,
  )
  .unwrap();
  assert_eq!(minimal.download_pattern(), "speaker_segmenter/*/*/*");
}

#[test]
fn model_info_accessors() {
  let info = ModelInfo::try_new(
    "speaker_segmenter",
    Some("pyannote-v3".to_string()),
    Some("W8A16".to_string()),
    coremlit::ComputeUnits::CpuAndNeuralEngine,
  )
  .unwrap();
  assert_eq!(info.name(), "speaker_segmenter");
  assert_eq!(info.version(), Some("pyannote-v3"));
  assert_eq!(info.variant(), Some("W8A16"));
  assert_eq!(info.compute(), coremlit::ComputeUnits::CpuAndNeuralEngine);

  let minimal = ModelInfo::try_new("x", None, None, coremlit::ComputeUnits::CpuOnly).unwrap();
  assert_eq!(minimal.version(), None);
  assert_eq!(minimal.variant(), None);
}

#[test]
fn model_info_find_base_folder_walks_up_to_name() {
  // ModelDownloaderTests.testModelInfoFindBaseFolder (ModelDownloaderTests.swift:220-227),
  // exercised against a real tempdir tree per this task's brief (Swift's own
  // version is pure path-component manipulation with no filesystem I/O at
  // all -- this port is hermetically stronger, not weaker).
  let info = ModelInfo::try_new(
    "speaker_segmenter",
    Some("pyannote-v3".to_string()),
    Some("W8A16".to_string()),
    coremlit::ComputeUnits::CpuAndNeuralEngine,
  )
  .unwrap();
  let dir = tempfile::tempdir().unwrap();
  let leaf = dir.path().join("speaker_segmenter/pyannote-v3/W8A16");
  std::fs::create_dir_all(&leaf).unwrap();

  let base = info.find_base_folder(&leaf).unwrap();
  assert_eq!(base, dir.path().to_path_buf());

  let elsewhere = dir.path().join("somewhere/else");
  std::fs::create_dir_all(&elsewhere).unwrap();
  assert_eq!(info.find_base_folder(&elsewhere), None);
}

// ---------------------------------------------------------------------
// SupportConfig / ModelSupport / DeviceSupport
// ---------------------------------------------------------------------

#[test]
fn support_config_longest_prefix_match() {
  let json = r#"{
    "name": "test-config",
    "version": "0.1",
    "device_support": [
      {
        "chips": "M2, M3, M4",
        "identifiers": ["Mac14"],
        "models": {
          "default": "generic-model",
          "supported": ["generic-model", "specific-model"]
        }
      },
      {
        "chips": "M2 Max",
        "identifiers": ["Mac14,13"],
        "models": {
          "default": "specific-model",
          "supported": ["specific-model"]
        }
      }
    ]
  }"#;
  let cfg = SupportConfig::from_json(json).unwrap();
  assert_eq!(
    cfg.support_for("Mac14,13").default_model(),
    "specific-model"
  );
  assert_eq!(cfg.support_for("Mac14,2").default_model(), "generic-model");
}

#[test]
fn support_config_unknown_device_falls_back_to_default_support() {
  let json = r#"{"name":"c","version":"1","device_support":[
    {"identifiers":["Mac14"],"models":{"default":"m","supported":["m"]}}
  ]}"#;
  let cfg = SupportConfig::from_json(json).unwrap();
  let fallback = cfg.support_for("totally-unknown-device");
  assert_eq!(fallback.default_model(), DEFAULT_FALLBACK_MODEL_NAME);
  assert_eq!(fallback.supported_slice().to_vec(), vec!["m".to_string()]);
}

#[test]
fn support_config_computes_disabled_as_known_minus_supported() {
  let json = r#"{"name":"c","version":"1","device_support":[
    {"identifiers":["A"],"models":{"default":"m1","supported":["m1"]}},
    {"identifiers":["B"],"models":{"default":"m2","supported":["m2"]}}
  ]}"#;
  let cfg = SupportConfig::from_json(json).unwrap();
  let a = cfg.support_for("A");
  assert_eq!(a.disabled_slice().to_vec(), vec!["m2".to_string()]);
  let b = cfg.support_for("B");
  assert_eq!(b.disabled_slice().to_vec(), vec!["m1".to_string()]);
}

#[test]
fn support_config_from_json_rejects_malformed_input() {
  assert!(SupportConfig::from_json("not json").is_err());
  assert!(SupportConfig::from_json(r#"{"name":"c"}"#).is_err()); // missing device_support
  assert!(SupportConfig::from_json(r#"{"device_support":[{"identifiers":["A"]}]}"#).is_err()); // missing models
}

#[test]
fn support_config_fallback_matches_swift_table() {
  // Constants.fallbackModelSupportConfig (Models.swift:1465-1662): 6 device
  // entries, extracted mechanically (see task-7 report for the script).
  let cfg = SupportConfig::fallback();
  assert_eq!(cfg.device_supports_slice().len(), 6);

  // A14 tier (iPhone13-class).
  let a14 = cfg.support_for("iPhone13,4");
  assert_eq!(a14.default_model(), "openai_whisper-base");
  assert!(
    a14
      .supported_slice()
      .iter()
      .any(|m| m == "openai_whisper-small.en")
  );

  // M2/M3/M4 tier.
  let apple_silicon = cfg.support_for("Mac14,2");
  assert_eq!(
    apple_silicon.default_model(),
    "openai_whisper-large-v3-v20240930"
  );
}

// ---------------------------------------------------------------------
// device_identifier
// ---------------------------------------------------------------------

#[test]
fn device_identifier_is_nonempty() {
  assert!(!device_identifier().is_empty());
}

// ---------------------------------------------------------------------
// ModelLoader / LocalModelLoader
// ---------------------------------------------------------------------

#[test]
fn local_loader_resolves_tiny_folder() {
  // hermetic: build a fake folder tree in tempdir with empty .mlmodelc dirs
  let dir = tempfile::tempdir().unwrap();
  for name in [
    "MelSpectrogram.mlmodelc",
    "AudioEncoder.mlmodelc",
    "TextDecoder.mlmodelc",
  ] {
    std::fs::create_dir_all(dir.path().join(name)).unwrap();
  }
  let resolved = LocalModelLoader::new().resolve(dir.path()).unwrap();
  assert!(resolved.mel_ref().ends_with("MelSpectrogram.mlmodelc"));
  assert!(resolved.encoder_ref().ends_with("AudioEncoder.mlmodelc"));
  assert!(resolved.decoder_ref().ends_with("TextDecoder.mlmodelc"));
}

#[test]
fn local_loader_errors_when_a_component_is_missing() {
  let dir = tempfile::tempdir().unwrap();
  std::fs::create_dir_all(dir.path().join("MelSpectrogram.mlmodelc")).unwrap();
  let err = LocalModelLoader::new().resolve(dir.path()).unwrap_err();
  assert!(matches!(err, ModelError::NotFound { .. }));
}

#[test]
fn local_model_loader_default_matches_new() {
  // A fieldless unit struct's derive(Default) cannot drift from `new()`;
  // clippy::default_constructed_unit_structs forbids spelling it
  // `::default()`, so compare the unit value against `new()` directly.
  assert_eq!(LocalModelLoader, LocalModelLoader::new());
}
