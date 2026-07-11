use std::path::Path;

use super::*;

#[test]
fn decoding_defaults_match_swift() {
  // Swift Configurations.swift DecodingOptions defaults (spec §6.2).
  let o = DecodingOptions::new();
  assert_eq!(o.task(), Task::Transcribe);
  assert!(o.language().is_empty()); // empty = auto-detect (golden empty-means-absent)
  assert_eq!(o.temperature(), 0.0);
  assert_eq!(o.temperature_increment_on_fallback(), 0.2);
  assert_eq!(o.temperature_fallback_count(), 5);
  assert_eq!(o.sample_length(), 224);
  assert_eq!(o.top_k(), 5);
  assert!(o.use_prefill_prompt());
  assert_eq!(o.use_prefill_prompt(), DEFAULT_USE_PREFILL_PROMPT); // pin to const
  assert!(!o.detect_language());
  assert!(!o.skip_special_tokens());
  assert!(!o.without_timestamps());
  assert!(!o.word_timestamps());
  assert_eq!(o.max_initial_timestamp(), None);
  assert_eq!(o.max_window_seek(), None);
  assert!(o.clip_timestamps_slice().is_empty());
  assert_eq!(o.window_clip_time(), 1.0);
  assert!(o.prompt_tokens_slice().is_empty());
  assert!(o.prefix_tokens_slice().is_empty());
  assert!(!o.suppress_blank());
  assert!(o.suppress_tokens_slice().is_empty());
  assert_eq!(o.compression_ratio_threshold(), Some(2.4));
  assert_eq!(o.logprob_threshold(), Some(-1.0));
  assert_eq!(o.first_token_logprob_threshold(), Some(-1.5));
  assert_eq!(o.no_speech_threshold(), Some(0.6));
  assert_eq!(o.concurrent_worker_count().get(), 16);
  assert_eq!(o.chunking_strategy(), ChunkingStrategy::Disabled);
  assert!(!o.verbose());
  assert_eq!(DecodingOptions::default(), DecodingOptions::new());
}

#[test]
fn builder_and_mutator_vocabulary() {
  let o = DecodingOptions::new()
    .with_temperature(0.4)
    .with_no_speech_threshold(0.9) // set_ = present value
    .maybe_logprob_threshold(None) // maybe_ = raw Option
    .with_without_timestamps(); // bool with_ takes no arg
  assert_eq!(o.temperature(), 0.4);
  assert_eq!(o.no_speech_threshold(), Some(0.9));
  assert_eq!(o.logprob_threshold(), None);
  assert!(o.without_timestamps());
  let mut m = DecodingOptions::new();
  m.set_top_k(10)
    .clear_compression_ratio_threshold()
    .set_detect_language();
  assert_eq!(m.top_k(), 10);
  assert_eq!(m.compression_ratio_threshold(), None);
  assert!(m.detect_language());
}

#[test]
fn enums_round_trip_and_display() {
  for t in [Task::Transcribe, Task::Translate] {
    assert_eq!(t.as_str().parse::<Task>().unwrap(), t);
  }
  assert_eq!(ChunkingStrategy::Vad.to_string(), "vad");
  assert_eq!(ChunkingStrategy::Disabled.as_str(), "none"); // wire parity, spec §6.1
  assert_eq!(
    "none".parse::<ChunkingStrategy>().unwrap(),
    ChunkingStrategy::Disabled
  );
  assert!("bogus".parse::<Task>().is_err());
}

#[cfg(feature = "serde")]
#[test]
fn serde_round_trips_and_fills_defaults() {
  let full = DecodingOptions::new().with_temperature(0.7);
  let json = serde_json::to_string(&full).unwrap();
  assert!(!json.contains("max_initial_timestamp")); // None skipped, golden §10
  assert_eq!(
    serde_json::from_str::<DecodingOptions>(&json).unwrap(),
    full
  );
  // partial config: everything missing falls back to new()
  let partial: DecodingOptions = serde_json::from_str(r#"{"temperature":0.5}"#).unwrap();
  assert_eq!(partial.temperature(), 0.5);
  assert_eq!(partial.top_k(), 5);
  assert_eq!(
    serde_json::from_str::<DecodingOptions>("{}").unwrap(),
    DecodingOptions::new()
  );
}

#[test]
fn compute_defaults_match_swift_model_compute_options() {
  let c = ComputeOptions::new();
  assert_eq!(c.mel(), coremlit::ComputeUnits::CpuAndGpu);
  assert_eq!(c.encoder(), coremlit::ComputeUnits::CpuAndNeuralEngine);
  assert_eq!(c.decoder(), coremlit::ComputeUnits::CpuAndNeuralEngine);
}

#[test]
fn task_predicates_display_and_default() {
  assert!(Task::Transcribe.is_transcribe());
  assert!(!Task::Transcribe.is_translate());
  assert!(Task::Translate.is_translate());
  assert_eq!(Task::default(), Task::Transcribe);
  assert_eq!(Task::Transcribe.to_string(), "transcribe");
}

#[test]
fn chunking_strategy_predicates_and_default() {
  assert!(ChunkingStrategy::Disabled.is_disabled());
  assert!(!ChunkingStrategy::Disabled.is_vad());
  assert!(ChunkingStrategy::Vad.is_vad());
  assert_eq!(ChunkingStrategy::default(), ChunkingStrategy::Disabled);
}

#[test]
fn parse_errors_are_opaque_and_display() {
  assert_eq!(
    "bogus".parse::<Task>().unwrap_err().to_string(),
    "unknown task name"
  );
  assert_eq!(
    "bogus".parse::<ChunkingStrategy>().unwrap_err().to_string(),
    "unknown chunking strategy name"
  );
}

#[test]
fn compute_options_builder_and_default() {
  let c = ComputeOptions::new()
    .with_mel(coremlit::ComputeUnits::CpuOnly)
    .with_encoder(coremlit::ComputeUnits::CpuOnly)
    .with_decoder(coremlit::ComputeUnits::CpuOnly);
  assert_eq!(c.mel(), coremlit::ComputeUnits::CpuOnly);
  assert_eq!(c.encoder(), coremlit::ComputeUnits::CpuOnly);
  assert_eq!(c.decoder(), coremlit::ComputeUnits::CpuOnly);
  assert_eq!(ComputeOptions::default(), ComputeOptions::new());
}

#[test]
fn options_new_requires_folders_and_defaults_rest() {
  let o = Options::new("/models/whisper", "/models/tokenizer");
  assert_eq!(o.model_folder(), Path::new("/models/whisper"));
  assert_eq!(o.tokenizer_folder(), Path::new("/models/tokenizer"));
  assert_eq!(o.compute(), ComputeOptions::new());
  assert!(!o.prewarm()); // Swift nil-prewarm resolves to "skip"
  assert!(o.load()); // Swift nil-load resolves to "load" when a folder is given
}

#[test]
fn options_builder_and_mutator_vocabulary() {
  let o = Options::new("a", "b")
    .with_prewarm()
    .with_compute(ComputeOptions::new().with_mel(coremlit::ComputeUnits::CpuOnly));
  assert!(o.prewarm());
  assert_eq!(o.compute().mel(), coremlit::ComputeUnits::CpuOnly);

  let mut m = Options::new("a", "b");
  m.clear_load();
  assert!(!m.load());
  m.update_load(true);
  assert!(m.load());
  m.set_prewarm();
  assert!(m.prewarm());
  m.clear_prewarm();
  assert!(!m.prewarm());
}

#[cfg(feature = "serde")]
#[test]
fn compute_options_serde_partial_uses_whisperkit_defaults() {
  // {} must resolve to WhisperKit's per-stage defaults, NOT
  // `ComputeUnits::default()` (`All`) — the field defaults are fn-backed,
  // not bare `#[serde(default)]`, precisely to avoid this.
  let c: ComputeOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(c, ComputeOptions::new());
  assert_ne!(c.mel(), coremlit::ComputeUnits::default());
  let json = serde_json::to_string(&ComputeOptions::new()).unwrap();
  assert_eq!(
    serde_json::from_str::<ComputeOptions>(&json).unwrap(),
    ComputeOptions::new()
  );
}

#[cfg(feature = "serde")]
#[test]
fn chunking_strategy_serde_renames_disabled_to_none() {
  assert_eq!(
    serde_json::to_string(&ChunkingStrategy::Disabled).unwrap(),
    "\"none\""
  );
  assert_eq!(
    serde_json::to_string(&ChunkingStrategy::Vad).unwrap(),
    "\"vad\""
  );
  assert_eq!(
    serde_json::from_str::<ChunkingStrategy>("\"none\"").unwrap(),
    ChunkingStrategy::Disabled
  );
}

#[cfg(feature = "serde")]
#[test]
fn task_serde_uses_snake_case() {
  assert_eq!(
    serde_json::to_string(&Task::Transcribe).unwrap(),
    "\"transcribe\""
  );
  assert_eq!(
    serde_json::from_str::<Task>("\"translate\"").unwrap(),
    Task::Translate
  );
}

#[cfg(feature = "serde")]
#[test]
fn options_serde_round_trips_and_fills_defaults() {
  let full = Options::new("/models/whisper", "/models/tokenizer").with_prewarm();
  let json = serde_json::to_string(&full).unwrap();
  assert_eq!(serde_json::from_str::<Options>(&json).unwrap(), full);

  // Partial config: only the required folders given; compute/prewarm/load
  // fall back to their defaults (load=true, NOT bool::default()=false).
  let partial: Options =
    serde_json::from_str(r#"{"model_folder":"/m","tokenizer_folder":"/t"}"#).unwrap();
  assert_eq!(partial.compute(), ComputeOptions::new());
  assert!(!partial.prewarm());
  assert!(partial.load());
}

#[cfg(feature = "serde")]
#[test]
fn decoding_options_empty_collections_skip_serializing() {
  let json = serde_json::to_string(&DecodingOptions::new()).unwrap();
  let value: serde_json::Value = serde_json::from_str(&json).unwrap();
  let object = value.as_object().unwrap();
  // Exact-key checks: `detect_language` (always serialized) contains the
  // substring "language", so a substring match on the JSON text would be a
  // false negative for the `language` field specifically.
  assert!(!object.contains_key("language"));
  assert!(!object.contains_key("clip_timestamps"));
  assert!(!object.contains_key("prompt_tokens"));
  assert!(!object.contains_key("prefix_tokens"));
  assert!(!object.contains_key("suppress_tokens"));
  assert!(object.contains_key("detect_language"));
  assert_eq!(
    serde_json::from_str::<DecodingOptions>(&json).unwrap(),
    DecodingOptions::new()
  );
}

#[cfg(feature = "serde")]
#[test]
fn compute_units_rejects_unknown_names() {
  let err = serde_json::from_str::<ComputeOptions>(r#"{"mel":"bogus"}"#).unwrap_err();
  assert!(err.to_string().contains("unknown compute units name"));
}
