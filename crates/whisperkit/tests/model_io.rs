//! Ground-truth introspection of the tiny model's encoder + decoder I/O.
//!
//! Values feed `whisperkit::backend`: Task 2 (`ModelDims` tiny defaults) and
//! Task 9 (`CoreMlBackend` feature names + `CoreMlDecoderState` allocation).
//! Swift-side derivations: `TextDecoder.swift:309-331` (dims read positions),
//! `Models.swift:970-1107` (generated I/O wrappers), `FeatureExtractor.swift:25-39`.

mod common;

use coremlit::{ComputeUnits, DataType, Model};

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn encoder_io_matches_swift_contract() {
  let model = Model::load(
    common::tiny_dir().join("AudioEncoder.mlmodelc"),
    ComputeUnits::CpuOnly,
  )
  .unwrap();
  let description = model.description();

  // Models.swift:909-931 AudioEncoderInput
  let input = description
    .input("melspectrogram_features")
    .expect("encoder input name");
  assert_eq!(input.shape(), &[1, 80, 1, 3000]);
  assert_eq!(input.data_type(), Some(DataType::F16));

  // Models.swift:934-960 AudioEncoderOutput; tiny embed dim = 384
  let output = description
    .output("encoder_output_embeds")
    .expect("encoder output name");
  assert_eq!(output.shape(), &[1, 384, 1, 1500]);
  assert_eq!(output.data_type(), Some(DataType::F16));
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn decoder_io_matches_swift_contract() {
  let model = Model::load(
    common::tiny_dir().join("TextDecoder.mlmodelc"),
    ComputeUnits::CpuOnly,
  )
  .unwrap();
  let description = model.description();

  // Inputs — names from TextDecoderInput (Models.swift:970-1035); shapes
  // from prepareDecoderInputs allocations (TextDecoder.swift:137-143) and
  // dim-read positions (TextDecoder.swift:317-331). tiny: kv_dim = 1536
  // (384 embed x 4 decoder layers), max context 224, audio ctx 1500.
  let input_ids = description.input("input_ids").expect("input_ids");
  assert_eq!(input_ids.shape(), &[1]);
  assert_eq!(input_ids.data_type(), Some(DataType::I32));

  let cache_length = description.input("cache_length").expect("cache_length");
  assert_eq!(cache_length.shape(), &[1]);
  assert_eq!(cache_length.data_type(), Some(DataType::I32));

  for name in ["key_cache", "value_cache"] {
    let feature = description.input(name).expect("kv cache input");
    assert_eq!(feature.shape(), &[1, 1536, 1, 224], "{name}");
    assert_eq!(feature.data_type(), Some(DataType::F16), "{name}");
  }

  // Swift's Rust port allocates this int32 (TextDecoder.swift:142), but the
  // compiled tiny model itself declares it float16.
  // introspected: I32 -> F16
  let update_mask = description
    .input("kv_cache_update_mask")
    .expect("kv_cache_update_mask");
  assert_eq!(update_mask.shape(), &[1, 224]);
  assert_eq!(update_mask.data_type(), Some(DataType::F16));

  let encoder_embeds = description
    .input("encoder_output_embeds")
    .expect("encoder_output_embeds");
  assert_eq!(encoder_embeds.shape(), &[1, 384, 1, 1500]);
  assert_eq!(encoder_embeds.data_type(), Some(DataType::F16));

  let padding_mask = description
    .input("decoder_key_padding_mask")
    .expect("decoder_key_padding_mask");
  assert_eq!(padding_mask.shape(), &[1, 224]);
  assert_eq!(padding_mask.data_type(), Some(DataType::F16));

  // Outputs — names from TextDecoderOutput (Models.swift:1037-1107).
  //
  // LOGITS SHAPE NOTE: the generated wrapper doc says `1 x vocab x 1 x 1`
  // (Models.swift:1041) but Swift reads logitsSize at shape position 2
  // (TextDecoder.swift:313-315) and the filters index logits as `[0, 0, v]`
  // (LogitsFilter.swift:18) with a `[1, 1, n]` fillLastDimension precondition
  // (MLMultiArrayExtensions.swift:90-99) — evidence pointed at [1, 1, 51865],
  // and introspection confirms it: the model declares [1, 1, 51865], not the
  // wrapper doc's [1, 51865, 1, 1].
  // introspected: shape pinned to [1, 1, 51865] (product-only check dropped
  // now that the exact layout is known; both the pipeline's shape-product
  // vocab derivation and a direct [1, 1, v] index remain valid against it).
  let logits = description.output("logits").expect("logits");
  assert_eq!(logits.shape(), &[1, 1, 51865]);
  assert_eq!(logits.data_type(), Some(DataType::F16));

  for name in ["key_cache_updates", "value_cache_updates"] {
    let feature = description.output(name).expect("kv update output");
    assert_eq!(feature.shape(), &[1, 1536, 1, 1], "{name}");
    assert_eq!(feature.data_type(), Some(DataType::F16), "{name}");
  }

  // Word-timestamp head: present on whisperkit-coreml conversions
  // (TextDecoder.swift:309-311 probes it; Models.swift:1077-1080 says [1, 1500]).
  // If ABSENT on this model generation, delete this block, record the absence,
  // and Task 9 sets supports_alignment accordingly.
  let alignment = description
    .output("alignment_heads_weights")
    .expect("alignment head output");
  assert_eq!(alignment.shape(), &[1, 1500]);
  assert_eq!(alignment.data_type(), Some(DataType::F16));
}

#[test]
#[ignore = "requires local tiny model (WHISPERKIT_TEST_MODELS)"]
fn fixture_wavs_are_16khz_mono() {
  let jfk = common::load_wav_mono_f32(&common::fixtures_dir().join("audio/jfk.wav"));
  // 11.0 s at 16 kHz (afinfo-verified at plan time).
  assert_eq!(jfk.len(), 176_000);
  assert!(jfk.iter().any(|s| s.abs() > 0.01), "fixture has signal");
}
