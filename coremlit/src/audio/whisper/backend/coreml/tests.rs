use super::*;

#[test]
fn names_match_recorded_ground_truth() {
  // Pins Task 1's introspected names as compile-visible constants.
  assert_eq!(names::LOGITS, "logits");
  assert_eq!(names::KEY_UPDATES, "key_cache_updates");
  assert_eq!(names::VALUE_UPDATES, "value_cache_updates");
  assert_eq!(names::ALIGNMENT, "alignment_heads_weights");
  assert_eq!(names::KV_UPDATE_MASK, "kv_cache_update_mask");
}

// ── The three load contracts ───────────────────────────────────────────────
//
// `model::contract`'s tests drive every CLAUSE of `check_load_contract`. What
// these drive is this backend's own three `LoadContract`s — the feature names,
// the element types, the geometry each stage requires of the previous one, and
// the state clause — against descriptions built with the same fixture
// machinery, so a mis-stated contract is caught here.

use crate::{AxisRange, FeatureInfo, model::RawShapeConstraint};

/// A fixed-shape multi-array feature, exactly as a plain coremltools export
/// reports one: raw type 2, its declared shape as the sole enumerated shape,
/// and `(d, 1)` on every axis. Every feature of every staged whisperkit model
/// reads this way.
fn fixed(name: &str, shape: &[usize], dtype: DataType) -> FeatureInfo {
  FeatureInfo::from_parts(
    name.to_string(),
    shape.to_vec(),
    Some(dtype),
    false,
    Some(RawShapeConstraint::new(
      2,
      vec![shape.to_vec()],
      shape.iter().map(|d| AxisRange::new(*d, 1)).collect(),
    )),
  )
}

/// A `RangeDims` multi-array feature; `shape` is the DEFAULT.
fn ranged(name: &str, shape: &[usize], dtype: DataType, ranges: &[AxisRange]) -> FeatureInfo {
  FeatureInfo::from_parts(
    name.to_string(),
    shape.to_vec(),
    Some(dtype),
    false,
    Some(RawShapeConstraint::new(3, Vec::new(), ranges.to_vec())),
  )
}

// The tiny conversion's numbers, read off the staged
// `Models/whisperkit-coreml/openai_whisper-tiny/*.mlmodelc` with
// `Model::load(..).description()`. Spelled here rather than imported: the
// backend reads every one of them off the artifact and tables none.
const TINY_WINDOW: usize = 480_000;
const TINY_MELS: usize = 80;
const TINY_MEL_FRAMES: usize = 3_000;
const TINY_EMBED: usize = 384;
const TINY_AUDIO_CTX: usize = 1_500;
const TINY_KV: usize = 1_536;
const TINY_CTX: usize = 224;
const TINY_VOCAB: usize = 51_865;

fn tiny_mel_description() -> ModelDescription {
  ModelDescription::from_parts(
    vec![fixed(names::AUDIO, &[TINY_WINDOW], DataType::F16)],
    vec![fixed(
      names::MEL,
      &[1, TINY_MELS, 1, TINY_MEL_FRAMES],
      DataType::F16,
    )],
    Vec::new(),
  )
}

fn tiny_encoder_description() -> ModelDescription {
  ModelDescription::from_parts(
    vec![fixed(
      names::MEL,
      &[1, TINY_MELS, 1, TINY_MEL_FRAMES],
      DataType::F16,
    )],
    vec![fixed(
      names::ENCODER,
      &[1, TINY_EMBED, 1, TINY_AUDIO_CTX],
      DataType::F16,
    )],
    Vec::new(),
  )
}

/// The tiny decoder, exactly as it reads back: seven inputs, four outputs, no
/// state. `mutate` gets the input and output vectors before assembly so a test
/// can change one feature and leave the rest alone.
fn tiny_decoder_description_with(
  mutate: impl FnOnce(&mut Vec<FeatureInfo>, &mut Vec<FeatureInfo>),
) -> ModelDescription {
  let mut inputs = vec![
    fixed(names::INPUT_IDS, &[1], DataType::I32),
    fixed(names::CACHE_LENGTH, &[1], DataType::I32),
    fixed(names::KEY_CACHE, &[1, TINY_KV, 1, TINY_CTX], DataType::F16),
    fixed(
      names::VALUE_CACHE,
      &[1, TINY_KV, 1, TINY_CTX],
      DataType::F16,
    ),
    fixed(names::KV_UPDATE_MASK, &[1, TINY_CTX], DataType::F16),
    fixed(
      names::ENCODER,
      &[1, TINY_EMBED, 1, TINY_AUDIO_CTX],
      DataType::F16,
    ),
    fixed(names::PADDING_MASK, &[1, TINY_CTX], DataType::F16),
  ];
  let mut outputs = vec![
    fixed(names::LOGITS, &[1, 1, TINY_VOCAB], DataType::F16),
    fixed(names::KEY_UPDATES, &[1, TINY_KV, 1, 1], DataType::F16),
    fixed(names::VALUE_UPDATES, &[1, TINY_KV, 1, 1], DataType::F16),
    fixed(names::ALIGNMENT, &[1, TINY_AUDIO_CTX], DataType::F16),
  ];
  mutate(&mut inputs, &mut outputs);
  ModelDescription::from_parts(inputs, outputs, Vec::new())
}

fn tiny_decoder_description() -> ModelDescription {
  tiny_decoder_description_with(|_, _| {})
}

/// The tiny decoder's contract, at the numbers the mel and encoder stages
/// above would have handed it.
fn tiny_decoder_contract(supports_alignment: bool) -> LoadContract {
  decoder_contract(
    TINY_EMBED,
    TINY_AUDIO_CTX,
    TINY_KV,
    TINY_CTX,
    supports_alignment,
  )
}

fn check(description: &ModelDescription, contract: &LoadContract) -> Result<(), BackendError> {
  crate::model::contract::check_load_contract(description, contract)
    .map_err(contract_violation("decoder"))
}

/// All three staged descriptions satisfy their contracts, and each stage's
/// contract is built from the previous stage's read-back.
#[test]
fn the_three_contracts_accept_the_staged_tiny_descriptions() {
  assert!(
    crate::model::contract::check_load_contract(&tiny_mel_description(), &mel_contract()).is_ok()
  );
  assert!(
    crate::model::contract::check_load_contract(
      &tiny_encoder_description(),
      &encoder_contract(TINY_MELS, TINY_MEL_FRAMES),
    )
    .is_ok()
  );
  assert_eq!(
    check(&tiny_decoder_description(), &tiny_decoder_contract(true)),
    Ok(())
  );
}

/// **FALSIFIER (red first).** Nothing at load compared a dtype anywhere in this
/// backend, so a decoder declaring `kv_cache_update_mask` as `int32` — which is
/// what Swift's own port allocates (`TextDecoder.swift:142`) — passed
/// construction. The mask buffer was then allocated at the DECLARED dtype and
/// the very next `fill_at::<f16>` rejected it, one layer away from the model
/// that caused it. It is a contract dtype now.
#[test]
fn the_decoder_contract_refuses_a_mistyped_kv_cache_update_mask() {
  let description = tiny_decoder_description_with(|inputs, _| {
    inputs[4] = fixed(names::KV_UPDATE_MASK, &[1, TINY_CTX], DataType::I32);
  });
  let err = check(&description, &tiny_decoder_contract(true)).unwrap_err();
  assert!(
    matches!(&err, BackendError::Contract(c)
      if c.model() == "decoder"
        && c.feature() == names::KV_UPDATE_MASK
        && c.expected() == "float16"
        && c.actual() == "int32"),
    "{err}"
  );
}

/// The same clause on the OTHER mask, which was equally unchecked: five of the
/// seven decoder inputs were never looked at at all.
#[test]
fn the_decoder_contract_refuses_a_mistyped_padding_mask() {
  let description = tiny_decoder_description_with(|inputs, _| {
    inputs[6] = fixed(names::PADDING_MASK, &[1, TINY_CTX], DataType::F32);
  });
  let err = check(&description, &tiny_decoder_contract(true)).unwrap_err();
  assert!(
    matches!(&err, BackendError::Contract(c) if c.feature() == names::PADDING_MASK),
    "{err}"
  );
}

/// **FALSIFIER (red first).** State is not an input — it lives in its own
/// dictionary and never appears among the ordinary inputs — so a decoder
/// declaring exactly these seven inputs PLUS a state buffer cleared every check
/// this backend made, and would then meet `decode_step`, which predicts through
/// the stateless API CoreML does not let a stateful model be called with.
///
/// No whisperkit artifact declares state (`StateContract::None`'s doc carries
/// that measurement, taken on all three sizes); this is the clause that keeps a
/// future one from arriving unnoticed.
#[test]
fn the_decoder_contract_refuses_a_decoder_that_declares_state() {
  let base = tiny_decoder_description();
  let description = ModelDescription::from_parts(
    base.inputs().to_vec(),
    base.outputs().to_vec(),
    vec![fixed("kv_cache", &[1, TINY_KV], DataType::F16)],
  );
  let err = check(&description, &tiny_decoder_contract(true)).unwrap_err();
  assert!(
    matches!(&err, BackendError::Contract(c)
      if c.feature() == "kv_cache" && c.actual() == "a declared state buffer"),
    "{err}"
  );
}

/// A `value_cache` that disagrees with `key_cache` about `kv_dim`. The decoder
/// state allocates BOTH at `key_cache`'s numbers, so this used to load and then
/// fail on the first prediction.
#[test]
fn the_decoder_contract_refuses_a_value_cache_that_disagrees_with_the_key_cache() {
  let description = tiny_decoder_description_with(|inputs, _| {
    inputs[3] = fixed(
      names::VALUE_CACHE,
      &[1, TINY_KV * 2, 1, TINY_CTX],
      DataType::F16,
    );
  });
  let err = check(&description, &tiny_decoder_contract(true)).unwrap_err();
  assert!(
    matches!(&err, BackendError::Contract(c) if c.feature() == names::VALUE_CACHE),
    "{err}"
  );
}

/// A mask narrower than `max_token_context`: the decode loop writes
/// `[0, position + 1]` up to `max_ctx - 1`, so this is an out-of-bounds write
/// waiting for the step that reaches it.
#[test]
fn the_decoder_contract_refuses_a_mask_narrower_than_the_kv_cache() {
  let description = tiny_decoder_description_with(|inputs, _| {
    inputs[4] = fixed(names::KV_UPDATE_MASK, &[1, TINY_CTX / 2], DataType::F16);
  });
  let err = check(&description, &tiny_decoder_contract(true)).unwrap_err();
  assert!(
    matches!(&err, BackendError::Contract(c) if c.feature() == names::KV_UPDATE_MASK),
    "{err}"
  );
}

/// **The stage-to-stage link.** A decoder built for a DIFFERENT encoder — the
/// small conversion's 768-wide embedding against tiny's 384 — is refused,
/// because the decoder's contract states `encoder_output_embeds` at the
/// encoder's own checked read-back rather than at a number of its own.
#[test]
fn the_decoder_contract_refuses_an_encoder_output_from_another_model_size() {
  let description = tiny_decoder_description_with(|inputs, _| {
    inputs[5] = fixed(names::ENCODER, &[1, 768, 1, TINY_AUDIO_CTX], DataType::F16);
  });
  let err = check(&description, &tiny_decoder_contract(true)).unwrap_err();
  assert!(
    matches!(&err, BackendError::Contract(c) if c.feature() == names::ENCODER),
    "{err}"
  );
}

/// The same link one stage earlier, on the model that had **no contract at
/// all**: the encoder's input must be the mel model's own output.
#[test]
fn the_encoder_contract_refuses_an_input_that_is_not_the_mels_output() {
  // A large-v3 encoder (128 mel bins) against a tiny mel model (80).
  let description = ModelDescription::from_parts(
    vec![fixed(
      names::MEL,
      &[1, 128, 1, TINY_MEL_FRAMES],
      DataType::F16,
    )],
    vec![fixed(
      names::ENCODER,
      &[1, 1280, 1, TINY_AUDIO_CTX],
      DataType::F16,
    )],
    Vec::new(),
  );
  let violation = crate::model::contract::check_load_contract(
    &description,
    &encoder_contract(TINY_MELS, TINY_MEL_FRAMES),
  )
  .unwrap_err();
  assert!(
    matches!(&violation, ContractViolation::Axis(a) if a.feature() == names::MEL),
    "{violation}"
  );
}

/// **The layout the shape PRODUCT could not see.** `vocab` was derived as the
/// product of the `logits` shape, so `[1, V, 1, 1]` — which the generated Swift
/// wrapper's own doc claims (`Models.swift:1041`) — was indistinguishable from
/// the `[1, 1, V]` every staged artifact declares and the filters index.
#[test]
fn the_decoder_contract_refuses_a_transposed_logits_head() {
  let description = tiny_decoder_description_with(|_, outputs| {
    outputs[0] = fixed(names::LOGITS, &[1, TINY_VOCAB, 1, 1], DataType::F16);
  });
  // The two layouts have the same shape PRODUCT, which is exactly why the
  // derivation this replaced could not tell them apart.
  assert_eq!(
    [1, TINY_VOCAB, 1, 1].iter().product::<usize>(),
    [1, 1, TINY_VOCAB].iter().product::<usize>()
  );
  let err = check(&description, &tiny_decoder_contract(true)).unwrap_err();
  assert!(
    matches!(&err, BackendError::Contract(c) if c.feature() == names::LOGITS),
    "{err}"
  );
}

/// A flexible `key_cache` declaring the artifact's exact numbers as its
/// DEFAULT. The two anchors this backend reads back are `Dim::AnyFixed`, which
/// requires the whole feature to be `ShapeConstraint::Fixed` — without that,
/// `kv_dim` and `max_token_context` would be bound from a default the graph
/// does not require, and every decoder buffer allocated at it.
#[test]
fn the_decoder_contract_refuses_a_flexible_key_cache() {
  let description = tiny_decoder_description_with(|inputs, _| {
    inputs[2] = ranged(
      names::KEY_CACHE,
      &[1, TINY_KV, 1, TINY_CTX],
      DataType::F16,
      &[
        AxisRange::new(1, 1),
        AxisRange::new(TINY_KV, 1),
        AxisRange::new(1, 1),
        AxisRange::inclusive(1, 448),
      ],
    );
  });
  let err = check(&description, &tiny_decoder_contract(true)).unwrap_err();
  assert!(
    matches!(&err, BackendError::Contract(c)
      if c.feature() == names::KEY_CACHE && c.expected() == "fixed" && c.actual() == "range"),
    "{err}"
  );
}

/// A REQUIRED input none of the seven names — the defect that fails every
/// prediction on a model that loaded clean.
#[test]
fn the_decoder_contract_refuses_a_required_input_it_does_not_name() {
  let description = tiny_decoder_description_with(|inputs, _| {
    inputs.push(fixed("prompt_ids", &[1, 16], DataType::I32));
  });
  let err = check(&description, &tiny_decoder_contract(true)).unwrap_err();
  assert!(
    matches!(&err, BackendError::Contract(c) if c.feature() == "prompt_ids"),
    "{err}"
  );
}

/// The alignment head is named only when the model declares it: a conversion
/// without the cross-attention head is legal and must still load, which is why
/// `supports_word_timestamps` reports it rather than the contract requiring it.
#[test]
fn a_decoder_without_the_alignment_head_is_accepted_when_the_contract_omits_it() {
  let description = tiny_decoder_description_with(|_, outputs| {
    outputs.pop();
  });
  assert_eq!(check(&description, &tiny_decoder_contract(false)), Ok(()));
  // And naming it against a model that lacks it is the refusal that keeps
  // `supports_alignment` honest.
  let err = check(&description, &tiny_decoder_contract(true)).unwrap_err();
  assert!(
    matches!(&err, BackendError::Contract(c)
      if c.feature() == names::ALIGNMENT && c.actual() == "missing"),
    "{err}"
  );
}

/// **The wiring, pinned on a REAL model, in every `cargo test`.**
///
/// Every other contract gate here drives `check_load_contract` over a fixture.
/// This one runs `CoreMlBackend::new` against
/// `Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc` — COMMITTED, 1.1
/// MiB, staged by no download — in all three positions, so unlike every other
/// gate that loads a whisper model it carries no `#[ignore]`.
///
/// Silero is a real, fixed-shape, six-feature CoreML graph that is simply not
/// any of these three models, which is the exact shape of a mis-pointed model
/// directory. Delete a `Checked::new` from `new` and this is the gate that
/// reds; the fixture gates above call the checker directly and would all still
/// pass.
#[test]
fn the_backend_contracts_refuse_the_vendored_silero_bundle() {
  let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc");
  assert!(
    bundle.is_dir(),
    "the vendored silero bundle is committed, so this gate is NOT model-gated; looked for {}",
    bundle.display()
  );
  let load = || Model::load(&bundle, crate::ComputeUnits::CpuOnly).expect("committed bundle loads");
  let err = CoreMlBackend::new(load(), load(), load()).expect_err("silero is not a whisper model");
  // The mel model is checked first, and silero declares no `audio` input.
  assert!(
    matches!(&err, BackendError::Contract(c)
      if c.model() == "mel" && c.feature() == names::AUDIO && c.actual() == "missing"),
    "{err}"
  );
}

/// `base` with one axis of one feature made one larger, rebuilt through the
/// same fixture constructor.
fn with_axis_bumped(base: &ModelDescription, feature: &str, axis: usize) -> ModelDescription {
  let bump = |declared: &FeatureInfo| -> FeatureInfo {
    if declared.name() != feature {
      return declared.clone();
    }
    let mut shape = declared.shape().to_vec();
    shape[axis] += 1;
    fixed(
      declared.name(),
      &shape,
      declared.data_type().expect("a multi-array feature"),
    )
  };
  ModelDescription::from_parts(
    base.inputs().iter().map(bump).collect(),
    base.outputs().iter().map(bump).collect(),
    base.states().to_vec(),
  )
}

/// `base` with one feature's element type changed to `dtype`.
fn with_dtype_changed(base: &ModelDescription, feature: &str, dtype: DataType) -> ModelDescription {
  let swap = |declared: &FeatureInfo| -> FeatureInfo {
    if declared.name() != feature {
      return declared.clone();
    }
    fixed(declared.name(), declared.shape(), dtype)
  };
  ModelDescription::from_parts(
    base.inputs().iter().map(swap).collect(),
    base.outputs().iter().map(swap).collect(),
    base.states().to_vec(),
  )
}

/// **Every axis clause across all three contracts is load-bearing, and the free
/// ones are named.**
///
/// One test per dimension is a list that silently stops covering an axis a
/// contract later gains — and on this door there are thirty-one of them across
/// three models. This perturbs EVERY axis of every named feature in turn and
/// asserts the contract refuses it, EXCEPT for the axes each stage deliberately
/// READS back off its checked model.
///
/// It reds in both directions: loosen a pinned axis and its perturbation is
/// wrongly accepted; pin a read-back axis and its perturbation is wrongly
/// refused.
/// One stage of the three-model chain, as the two sweeps below drive it.
struct Stage {
  /// `mel`, `encoder` or `decoder` — the name a failure is reported under.
  name: &'static str,
  /// That stage's contract, at the numbers the previous stage read back.
  contract: LoadContract,
  /// The staged tiny conversion's description for that model.
  description: ModelDescription,
  /// The `(feature, axis)` pairs this stage READS back rather than requires —
  /// every other axis of every named feature is pinned.
  free_axes: &'static [(&'static str, usize)],
}

/// The three stages, in the order `CoreMlBackend::new` checks them.
fn contract_stages() -> [Stage; 3] {
  [
    Stage {
      name: "mel",
      contract: mel_contract(),
      description: tiny_mel_description(),
      // window_samples; then n_mels and the mel frame count.
      free_axes: &[(names::AUDIO, 0), (names::MEL, 1), (names::MEL, 3)],
    },
    Stage {
      name: "encoder",
      contract: encoder_contract(TINY_MELS, TINY_MEL_FRAMES),
      description: tiny_encoder_description(),
      // embed_dim and n_audio_ctx.
      free_axes: &[(names::ENCODER, 1), (names::ENCODER, 3)],
    },
    Stage {
      name: "decoder",
      contract: tiny_decoder_contract(true),
      description: tiny_decoder_description(),
      // kv_dim and max_token_context off `key_cache`, and vocab off `logits`.
      free_axes: &[
        (names::KEY_CACHE, 1),
        (names::KEY_CACHE, 3),
        (names::LOGITS, 2),
      ],
    },
  ]
}

#[test]
fn every_axis_is_pinned_except_the_dimensions_each_stage_reads_back() {
  let mut perturbations = 0_usize;
  for stage in contract_stages() {
    let base = &stage.description;
    for declared in base.inputs().iter().chain(base.outputs()) {
      for axis in 0..declared.shape().len() {
        let perturbed = with_axis_bumped(base, declared.name(), axis);
        let free = stage.free_axes.contains(&(declared.name(), axis));
        let accepted =
          crate::model::contract::check_load_contract(&perturbed, &stage.contract).is_ok();
        assert_eq!(
          accepted,
          free,
          "{}: `{}` axis {axis}: the contract {} it",
          stage.name,
          declared.name(),
          if free { "must accept" } else { "must refuse" }
        );
        perturbations += 1;
      }
    }
  }
  // Non-vacuous: mel 1 + 4, encoder 4 + 4, decoder 1 + 1 + 4 + 4 + 2 + 4 + 2
  // inputs and 3 + 4 + 4 + 2 outputs.
  assert_eq!(perturbations, 44);
}

/// **Every dtype clause is load-bearing too**, and it is the clause this door
/// had none of: nothing at load compared an element type anywhere. Each named
/// feature in turn is re-declared at a type the door does not write, and every
/// one must be refused.
#[test]
fn every_named_features_element_type_is_pinned() {
  let mut checked = 0_usize;
  for stage in contract_stages() {
    let base = &stage.description;
    let names: Vec<String> = base
      .inputs()
      .iter()
      .chain(base.outputs())
      .map(|f| f.name().to_string())
      .collect();
    for name in names {
      let declared = base
        .input(&name)
        .or_else(|| base.output(&name))
        .expect("just enumerated");
      // Any type the contract does not state for this feature.
      let other = if declared.data_type() == Some(DataType::I32) {
        DataType::F16
      } else {
        DataType::I32
      };
      let perturbed = with_dtype_changed(base, &name, other);
      assert!(
        crate::model::contract::check_load_contract(&perturbed, &stage.contract).is_err(),
        "{}: `{name}` re-declared {other:?} must be refused",
        stage.name
      );
      checked += 1;
    }
  }
  // 2 mel + 2 encoder + 11 decoder features.
  assert_eq!(checked, 15);
}

/// `base` with one axis of one feature set to `size`.
fn with_axis_set(
  base: &ModelDescription,
  feature: &str,
  axis: usize,
  size: usize,
) -> ModelDescription {
  let set = |declared: &FeatureInfo| -> FeatureInfo {
    if declared.name() != feature {
      return declared.clone();
    }
    let mut shape = declared.shape().to_vec();
    shape[axis] = size;
    fixed(
      declared.name(),
      &shape,
      declared.data_type().expect("a multi-array feature"),
    )
  };
  ModelDescription::from_parts(
    base.inputs().iter().map(set).collect(),
    base.outputs().iter().map(set).collect(),
    base.states().to_vec(),
  )
}

/// **Every read-back axis carries a FLOOR**, which is the half the sweep above
/// cannot see: it perturbs a free axis UPWARD, and a free axis is meant to
/// accept that.
///
/// A zero-sized read-back axis is pinned — it admits exactly one size, and that
/// size is `0` — so it satisfies `Dim::AnyFixed` and only the floor refuses it.
/// Every one of these numbers is then allocated from: `window_samples` sizes
/// the audio window, `n_mels`/`n_audio_ctx` the scratch buffers, `kv_dim` and
/// `max_token_context` the KV caches and both masks, `vocab` the logits gather.
/// A zero in any of them is a graph that loads and produces nothing.
#[test]
fn every_read_back_axis_refuses_a_zero_size() {
  let mut floors = 0_usize;
  for stage in contract_stages() {
    for (feature, axis) in stage.free_axes {
      let zeroed = with_axis_set(&stage.description, feature, *axis, 0);
      assert!(
        crate::model::contract::check_load_contract(&zeroed, &stage.contract).is_err(),
        "{}: `{feature}` axis {axis} at size 0 must be refused",
        stage.name
      );
      floors += 1;
    }
  }
  // Non-vacuous: 3 mel + 2 encoder + 3 decoder read-back axes.
  assert_eq!(floors, 8);
}
