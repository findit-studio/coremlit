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
  assert_eq!(o.seed(), None); // Rust-only addition (coremlit issue #9): unset by default
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
  // Rust-only addition (coremlit issue #14), and the ONE default here that
  // deliberately does NOT match Swift: Swift emits `[BLANK_AUDIO]` for
  // silence, this drops it unless the caller opts back in. Pinned in full
  // by `drop_blank_audio_defaults_on_and_opts_out_to_swift_parity`.
  assert!(o.drop_blank_audio());
  // Rust-only addition (coremlit issue #14). Swift has no grouping knob at
  // all — it picks one from a language it detects internally, landing on
  // the coarse `Phrase` shape for CJK. This default preserves THIS port's
  // #11-pinned fine-grained CJK grouping; Swift's is the explicit opt-in.
  assert_eq!(o.word_grouping(), WordGrouping::FineGrained);
  assert_eq!(DecodingOptions::default(), DecodingOptions::new());
}

#[test]
fn drop_blank_audio_defaults_on_and_opts_out_to_swift_parity() {
  // Default ON (the maintainer decision on coremlit issue #14 — the
  // inverse of that issue's originally-proposed `suppress_blank_audio:
  // false`), pinned to the const so the two cannot drift apart.
  let o = DecodingOptions::new();
  assert!(o.drop_blank_audio());
  assert_eq!(o.drop_blank_audio(), DEFAULT_DROP_BLANK_AUDIO); // pin to const

  // Full bool vocabulary: `clear_` is the Swift-parity escape hatch.
  assert!(
    !DecodingOptions::new()
      .maybe_drop_blank_audio(false)
      .drop_blank_audio()
  );
  assert!(
    DecodingOptions::new()
      .with_drop_blank_audio()
      .drop_blank_audio()
  );
  let mut m = DecodingOptions::new();
  m.clear_drop_blank_audio();
  assert!(!m.drop_blank_audio(), "clear_ opts out to Swift parity");
  m.set_drop_blank_audio();
  assert!(m.drop_blank_audio());
  m.update_drop_blank_audio(false);
  assert!(!m.drop_blank_audio());
}

#[cfg(feature = "serde")]
#[test]
fn drop_blank_audio_serde_default_is_true_not_bool_default() {
  // The whole reason this field carries `serde(default = "fn")` rather than
  // the bare `serde(default)`: `bool::default()` is `false`, which is the
  // OPPOSITE of this knob's contract. A config that omits the field must
  // still DROP; only an explicit `false` opts out.
  let omitted: DecodingOptions = serde_json::from_str("{}").unwrap();
  assert!(
    omitted.drop_blank_audio(),
    "an omitted field must default to dropping, not to bool::default()"
  );
  let explicit_false: DecodingOptions =
    serde_json::from_str(r#"{"drop_blank_audio":false}"#).unwrap();
  assert!(!explicit_false.drop_blank_audio());

  // Round-trips in both states (always serialized — never skipped, so a
  // persisted provenance-grade config always records which way it ran).
  for wanted in [true, false] {
    let options = DecodingOptions::new().maybe_drop_blank_audio(wanted);
    let json = serde_json::to_string(&options).unwrap();
    assert!(json.contains("drop_blank_audio"));
    assert_eq!(
      serde_json::from_str::<DecodingOptions>(&json).unwrap(),
      options
    );
  }
}

#[test]
fn builder_and_mutator_vocabulary() {
  let o = DecodingOptions::new()
    .with_temperature(0.4)
    .with_no_speech_threshold(0.9) // set_ = present value
    .maybe_logprob_threshold(None) // maybe_ = raw Option
    .with_without_timestamps() // bool with_ takes no arg
    .with_seed(7);
  assert_eq!(o.temperature(), 0.4);
  assert_eq!(o.no_speech_threshold(), Some(0.9));
  assert_eq!(o.logprob_threshold(), None);
  assert!(o.without_timestamps());
  assert_eq!(o.seed(), Some(7));
  let mut m = DecodingOptions::new();
  m.set_top_k(10)
    .clear_compression_ratio_threshold()
    .set_detect_language()
    .set_seed(11);
  assert_eq!(m.top_k(), 10);
  assert_eq!(m.compression_ratio_threshold(), None);
  assert!(m.detect_language());
  assert_eq!(m.seed(), Some(11));
  m.update_seed(Some(12));
  assert_eq!(m.seed(), Some(12));
  m.clear_seed();
  assert_eq!(m.seed(), None);
  let via_maybe = DecodingOptions::new().maybe_seed(Some(13));
  assert_eq!(via_maybe.seed(), Some(13));
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

  for g in [WordGrouping::FineGrained, WordGrouping::SwiftParity] {
    assert_eq!(g.as_str().parse::<WordGrouping>().unwrap(), g);
  }
  assert_eq!(WordGrouping::FineGrained.to_string(), "fine_grained");
  assert_eq!(WordGrouping::SwiftParity.as_str(), "swift_parity");
  assert!("bogus".parse::<WordGrouping>().is_err());
}

#[test]
fn word_grouping_defaults_to_fine_grained() {
  // The #11-pinned CJK behavior stays the default: a caller who never
  // mentions grouping gets the fine-grained split, and Swift's coarse
  // phrase-blob grouping is reachable only by naming it (coremlit #14).
  assert_eq!(WordGrouping::default(), WordGrouping::FineGrained);
  assert_eq!(
    DecodingOptions::new().word_grouping(),
    WordGrouping::FineGrained
  );
  assert!(DecodingOptions::new().word_grouping().is_fine_grained());

  let built = DecodingOptions::new().with_word_grouping(WordGrouping::SwiftParity);
  assert_eq!(built.word_grouping(), WordGrouping::SwiftParity);
  assert!(built.word_grouping().is_swift_parity());

  let mut m = DecodingOptions::new();
  m.set_word_grouping(WordGrouping::SwiftParity);
  assert_eq!(m.word_grouping(), WordGrouping::SwiftParity);
  m.set_word_grouping(WordGrouping::FineGrained);
  assert_eq!(m.word_grouping(), WordGrouping::FineGrained);
}

#[cfg(feature = "serde")]
#[test]
fn word_grouping_serde_omitted_stays_fine_grained() {
  // An omitted field must NOT silently opt a config into Swift's coarse
  // grouping — the whole point of making `Phrase` explicit.
  let omitted: DecodingOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(omitted.word_grouping(), WordGrouping::FineGrained);

  let phrase: DecodingOptions =
    serde_json::from_str(r#"{"word_grouping":"swift_parity"}"#).unwrap();
  assert_eq!(phrase.word_grouping(), WordGrouping::SwiftParity);

  for wanted in [WordGrouping::FineGrained, WordGrouping::SwiftParity] {
    let options = DecodingOptions::new().with_word_grouping(wanted);
    let json = serde_json::to_string(&options).unwrap();
    assert_eq!(
      serde_json::from_str::<DecodingOptions>(&json).unwrap(),
      options
    );
  }
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
fn detect_language_default_couples_to_prefill() {
  // Ports Swift's `detectLanguage ?? !usePrefillPrompt`
  // (Configurations.swift:222): unset detection defaults ON exactly when
  // prefill is OFF, and an explicit choice always wins (coremlit
  // issue #9 follow-up — this was the one genuine defaults gap between
  // the ports).

  // Unset + prefill ON (both defaults): resolves false — the pre-coupling
  // default path, byte-unchanged.
  let unset = DecodingOptions::new();
  assert!(unset.use_prefill_prompt());
  assert!(!unset.detect_language());
  assert_eq!(unset.detect_language, None, "constructed unset, not false");

  // Unset + prefill OFF, reached by an in-place mutator on an
  // already-constructed value: resolves true, this port's own
  // live-resolution rule — not Swift-identical. See
  // `DecodingOptions::detect_language`'s doc ("Pinned deviation") and
  // `detect_language_pinned_construction_vs_mutation_histories` below
  // for the construction-vs-mutation distinction Swift parity actually
  // depends on here.
  let mut no_prefill = DecodingOptions::new();
  no_prefill.clear_use_prefill_prompt();
  assert!(no_prefill.detect_language());

  // Explicit false beats the coupling even with prefill off...
  let mut explicit_false = DecodingOptions::new();
  explicit_false
    .clear_use_prefill_prompt()
    .clear_detect_language();
  assert!(!explicit_false.detect_language());

  // ...and mutation ORDER does not matter: the coupling is resolved at
  // read time, so flipping prefill after the explicit choice cannot
  // overwrite it (the construction-time-resolution failure mode).
  let mut late_prefill = DecodingOptions::new();
  late_prefill.clear_detect_language();
  late_prefill.clear_use_prefill_prompt();
  assert!(!late_prefill.detect_language());

  // Explicit true with prefill ON (coupling would say false).
  let explicit_true = DecodingOptions::new().with_detect_language();
  assert!(explicit_true.use_prefill_prompt());
  assert!(explicit_true.detect_language());

  // maybe_/update_ record an explicit choice too.
  let via_update = DecodingOptions::new()
    .maybe_use_prefill_prompt(false)
    .maybe_detect_language(false);
  assert!(!via_update.detect_language());
}

#[test]
fn detect_language_pinned_construction_vs_mutation_histories() {
  // `DecodingOptions::detect_language`'s doc names two histories for an
  // unset `detect_language` ending with `use_prefill_prompt == false`.
  // Both resolve to the SAME Rust value (`true`) — what differs is
  // which Swift call each history actually corresponds to.

  // (i) Construction: chain `with_*`/`maybe_*` from `new()`, never
  // mutate again. Swift-identical — matches
  // `DecodingOptions(usePrefillPrompt: false)`, whose initializer
  // computes `detectLanguage = nil ?? !false = true` directly from the
  // argument: same formula, same final inputs, same value.
  let constructed = DecodingOptions::new().maybe_use_prefill_prompt(false);
  assert_eq!(constructed.detect_language, None, "still unset");
  assert!(constructed.detect_language());

  // (ii) In-place mutation of an already-constructed value: the pinned
  // deviation. Also resolves `true` here, but Swift's equivalent is
  // `var o = DecodingOptions()` — `usePrefillPrompt` defaults `true`,
  // so `detectLanguage` freezes `!true == false` right there in `init`
  // — followed by the LATER, separate mutation `o.usePrefillPrompt =
  // false`, which cannot reach back and change the already-frozen
  // `detectLanguage`. Swift stays `false`; this port resolves `true`.
  // Kept deliberately (see the doc's tri-state/serde argument), not
  // "fixed" to match.
  let mut mutated = DecodingOptions::new();
  assert!(mutated.use_prefill_prompt()); // prefill ON — Swift's default init
  mutated.clear_use_prefill_prompt(); // separate, later mutation: Swift's `o.usePrefillPrompt = false`
  assert_eq!(mutated.detect_language, None, "still unset");
  assert!(
    mutated.detect_language(),
    "pinned deviation: Swift's equivalent history stays false here"
  );

  // Both histories land on the identical Rust value, which is exactly
  // why this is a getter-shape (Swift-comparison-point) deviation
  // rather than a value bug reachable from a single Rust run.
  assert_eq!(constructed.detect_language(), mutated.detect_language());
}

#[cfg(feature = "serde")]
#[test]
fn detect_language_serde_tristate() {
  // (d) of the coupling contract: unset serializes ABSENT and a missing
  // field deserializes as unset (still coupled), while explicit values
  // round-trip and win.

  // Unset -> key absent (also covered by the exact-key test above).
  let json = serde_json::to_string(&DecodingOptions::new()).unwrap();
  let value: serde_json::Value = serde_json::from_str(&json).unwrap();
  assert!(!value.as_object().unwrap().contains_key("detect_language"));

  // Missing field -> unset, not Some(false): with prefill also absent
  // (default ON) it resolves false...
  let missing: DecodingOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(missing.detect_language, None);
  assert!(!missing.detect_language());
  // ...and with prefill explicitly OFF in the config, the still-unset
  // field resolves true — a partial config gets the Swift coupling.
  let coupled: DecodingOptions = serde_json::from_str(r#"{"use_prefill_prompt":false}"#).unwrap();
  assert_eq!(coupled.detect_language, None);
  assert!(coupled.detect_language());

  // Explicit false survives the round trip and beats the coupling.
  let explicit_false: DecodingOptions =
    serde_json::from_str(r#"{"use_prefill_prompt":false,"detect_language":false}"#).unwrap();
  assert_eq!(explicit_false.detect_language, Some(false));
  assert!(!explicit_false.detect_language());
  let json = serde_json::to_string(&explicit_false).unwrap();
  assert!(json.contains("\"detect_language\":false"));
  assert_eq!(
    serde_json::from_str::<DecodingOptions>(&json).unwrap(),
    explicit_false
  );

  // Explicit true round-trips as a present key.
  let explicit_true = DecodingOptions::new().with_detect_language();
  let json = serde_json::to_string(&explicit_true).unwrap();
  assert!(json.contains("\"detect_language\":true"));
  assert_eq!(
    serde_json::from_str::<DecodingOptions>(&json).unwrap(),
    explicit_true
  );
}

#[test]
fn compute_defaults_match_swift_model_compute_options() {
  // Pinned against the Swift source, `Models.swift:92-118`
  // (`ModelComputeOptions.init`):
  //
  //   melCompute:         MLComputeUnits = .cpuAndGPU
  //   audioEncoderCompute: MLComputeUnits? = nil
  //                        -> .cpuAndNeuralEngine  (macOS 14+/iOS 17+)
  //                        -> .cpuAndGPU           (older, not a target here)
  //   textDecoderCompute:  MLComputeUnits = .cpuAndNeuralEngine
  //
  // (Swift's `isRunningOnSimulator` branch forces all three to `.cpuOnly`;
  // this crate is macOS/CoreML-on-device only, so that branch has no
  // counterpart and is deliberately not ported.)
  //
  // These three constants are what the crate SHIPS, and therefore what the
  // parity goldens must run on: `jfk_tiny_golden.json`/`es_tiny_golden.json`
  // were captured from `whisperkit-cli @ argmax-oss-swift` on the ANE, so
  // `tests/parity_jfk.rs`/`parity_es.rs` assert `Options::new`'s compute
  // units against exactly these values before building the pipeline. A gate
  // validating a shipping default must run on the shipping default — see
  // those tests, and `tests/pipeline.rs`, for the rule. Changing a value
  // here without re-capturing the goldens against the new compute path is a
  // silent parity break; that is what this test exists to stop.
  assert_eq!(DEFAULT_MEL_COMPUTE_UNITS, crate::ComputeUnits::CpuAndGpu);
  assert_eq!(
    DEFAULT_ENCODER_COMPUTE_UNITS,
    crate::ComputeUnits::CpuAndNeuralEngine
  );
  assert_eq!(
    DEFAULT_DECODER_COMPUTE_UNITS,
    crate::ComputeUnits::CpuAndNeuralEngine
  );

  // ...and `ComputeOptions::new()` — what `Options::new` hands the pipeline —
  // really is built from those constants, not from `ComputeUnits::default()`
  // (which is `All`, coremlit's own general-purpose default, and matches
  // none of them).
  let c = ComputeOptions::new();
  assert_eq!(c.mel(), DEFAULT_MEL_COMPUTE_UNITS);
  assert_eq!(c.encoder(), DEFAULT_ENCODER_COMPUTE_UNITS);
  assert_eq!(c.decoder(), DEFAULT_DECODER_COMPUTE_UNITS);
  assert_eq!(Options::new("m", "t").compute(), c);
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
    .with_mel(crate::ComputeUnits::CpuOnly)
    .with_encoder(crate::ComputeUnits::CpuOnly)
    .with_decoder(crate::ComputeUnits::CpuOnly);
  assert_eq!(c.mel(), crate::ComputeUnits::CpuOnly);
  assert_eq!(c.encoder(), crate::ComputeUnits::CpuOnly);
  assert_eq!(c.decoder(), crate::ComputeUnits::CpuOnly);
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
    .with_compute(ComputeOptions::new().with_mel(crate::ComputeUnits::CpuOnly));
  assert!(o.prewarm());
  assert_eq!(o.compute().mel(), crate::ComputeUnits::CpuOnly);

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
  assert_ne!(c.mel(), crate::ComputeUnits::default());
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
  // Exact-key checks: an explicitly-set `detect_language` key contains
  // the substring "language", so a substring match on the JSON text would
  // be a false negative for the `language` field specifically.
  assert!(!object.contains_key("language"));
  assert!(!object.contains_key("clip_timestamps"));
  assert!(!object.contains_key("prompt_tokens"));
  assert!(!object.contains_key("prefix_tokens"));
  assert!(!object.contains_key("suppress_tokens"));
  // Unset tri-state `detect_language` is skipped like the `None` floats
  // (see `detect_language_serde_tristate` for the full presence matrix).
  assert!(!object.contains_key("detect_language"));
  // Unset `seed` (coremlit issue #9) follows the identical
  // `Option<Copy>` presence rule as `max_initial_timestamp`/
  // `max_window_seek` (see `seed_serde_absent_when_unset_round_trips_when_set`).
  assert!(!object.contains_key("seed"));
  assert_eq!(
    serde_json::from_str::<DecodingOptions>(&json).unwrap(),
    DecodingOptions::new()
  );
}

#[cfg(feature = "serde")]
#[test]
fn seed_serde_absent_when_unset_round_trips_when_set() {
  // Unset -> key absent (also covered by the exact-key test above).
  let json = serde_json::to_string(&DecodingOptions::new()).unwrap();
  let value: serde_json::Value = serde_json::from_str(&json).unwrap();
  assert!(!value.as_object().unwrap().contains_key("seed"));
  assert_eq!(
    serde_json::from_str::<DecodingOptions>(&json)
      .unwrap()
      .seed(),
    None
  );

  // Missing field -> None, not some sentinel value.
  let missing: DecodingOptions = serde_json::from_str("{}").unwrap();
  assert_eq!(missing.seed(), None);

  // Explicit value round-trips exactly, including through the full
  // u64 range (serde_json represents u64 natively, no float-precision
  // loss to worry about like it would for e.g. an f64 seed).
  for seed in [0u64, 1, 42, u64::MAX] {
    let explicit = DecodingOptions::new().with_seed(seed);
    let json = serde_json::to_string(&explicit).unwrap();
    assert!(json.contains(&format!("\"seed\":{seed}")));
    assert_eq!(
      serde_json::from_str::<DecodingOptions>(&json).unwrap(),
      explicit
    );
  }
}

#[cfg(feature = "serde")]
#[test]
fn compute_units_rejects_unknown_names() {
  let err = serde_json::from_str::<ComputeOptions>(r#"{"mel":"bogus"}"#).unwrap_err();
  assert!(err.to_string().contains("unknown compute units name"));
}

#[cfg(feature = "serde")]
#[test]
fn non_finite_floats_are_rejected_at_the_serde_boundary() {
  // Codex round 3, F6. `serde_json` has no JSON form for `NaN`/±∞ and silently
  // writes `null` for each, so `with_compression_ratio_threshold(-inf)` would
  // serialize `Some(-inf)` as `null` and deserialize back as `None` — a check
  // that fired on every finite ratio SILENTLY DISABLED across a round trip.
  // Swift closes the same hole from the encode side (`DecodingOptions: Codable`,
  // `Configurations.swift:155`, default throwing `JSONEncoder`); this port
  // refuses a non-finite float on BOTH sides of the wire, so the lossy `null`
  // is never produced and the round trip stays lossless. Covers all three field
  // shapes the finding names — scalar, optional, vector — against NaN and both
  // infinities.
  for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
    // scalar f32
    assert!(
      serde_json::to_string(&DecodingOptions::new().with_temperature(bad)).is_err(),
      "a non-finite scalar `temperature` must be refused, not written as null"
    );
    assert!(serde_json::to_string(&DecodingOptions::new().with_window_clip_time(bad)).is_err());
    // optional Option<f32> — the exact finding field
    assert!(
      serde_json::to_string(&DecodingOptions::new().with_compression_ratio_threshold(bad)).is_err(),
      "a non-finite threshold must be refused, or it round-trips to a forged `None`"
    );
    // vector Vec<f32>
    assert!(
      serde_json::to_string(&DecodingOptions::new().with_clip_timestamps(vec![0.0, bad])).is_err()
    );
  }

  // The deserialize side is guarded too: a JSON literal that overflows to a
  // non-finite value must be rejected, not silently accepted (serde_json parses
  // `1e400` to an infinity, and the finite guard then refuses it).
  assert!(serde_json::from_str::<DecodingOptions>(r#"{"temperature":1e400}"#).is_err());
  assert!(serde_json::from_str::<DecodingOptions>(r#"{"clip_timestamps":[1e400]}"#).is_err());
  assert!(
    serde_json::from_str::<DecodingOptions>(r#"{"compression_ratio_threshold":1e400}"#).is_err()
  );

  // `null` for a threshold is STILL the honest "check disabled" — only a
  // non-finite NUMBER is rejected, never a genuine `None`. This is the
  // distinction the whole finding turns on.
  let disabled: DecodingOptions =
    serde_json::from_str(r#"{"compression_ratio_threshold":null}"#).unwrap();
  assert_eq!(disabled.compression_ratio_threshold(), None);

  // A finite config — negative temperatures included, which F1 keeps valid
  // in-memory — still round-trips losslessly, the property the guard protects.
  let finite = DecodingOptions::new()
    .with_temperature(-0.2)
    .with_window_clip_time(0.5)
    .with_compression_ratio_threshold(2.4)
    .maybe_logprob_threshold(None)
    .with_clip_timestamps(vec![0.0, 1.5, 3.0]);
  let json = serde_json::to_string(&finite).unwrap();
  assert_eq!(
    serde_json::from_str::<DecodingOptions>(&json).unwrap(),
    finite
  );
}
