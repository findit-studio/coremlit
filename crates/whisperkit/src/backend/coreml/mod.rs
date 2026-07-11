//! [`CoreMlBackend`]: the real [`InferenceBackend`] over the three CoreML
//! models of a Whisper pipeline — `MelSpectrogram`, `AudioEncoder`,
//! `TextDecoder` (spec §5.4). Ports the model-facing halves of
//! `FeatureExtractor.swift:25-57` (mel), `AudioEncoder` prediction, and
//! `TextDecoder.swift` — dims-from-description (`:309-331`), the
//! `DecodingInputs` tensor set (`Models.swift:291-323`, allocation
//! `TextDecoder.swift:137-147`), the per-step input writes (`:600-602`),
//! `updateKVCache` (`:218-270`), the mask flips (`:704-707`),
//! `updateAlignmentWeights` (`:272-296`), and `DecodingInputs.reset`
//! (`Models.swift:312-322`).
//!
//! Tensor names/shapes/dtypes are the tiny model's introspected ground
//! truth, pinned by `tests/model_io.rs` (Task 1); the private `names`
//! module holds the feature names. Notable recorded deviation from the
//! Swift source: the compiled model declares `kv_cache_update_mask` as
//! `float16` even though Swift allocates it `int32`
//! (`TextDecoder.swift:142`) — allocation here follows the model's
//! declared dtype.
//!
//! Real prediction outputs can be row-padded (IOSurface-backed; e.g.
//! strides `[240640, 3008, 3008, 1]` for the mel output), which
//! `MultiArray::as_slice` refuses as non-contiguous — every model-output
//! extraction therefore goes through `MultiArray::copy_into`, which
//! gathers padded rows correctly. Tensors that only *flow between models*
//! (mel features, encoder output) are never read on the CPU at all and
//! stay owned `MultiArray`s end to end.

use coremlit::{DataType, Model, MultiArray, TensorError, f16};

use crate::{
  backend::{AlignmentView, BackendError, InferenceBackend, ModelDims},
  model::manager::LoadedModels,
};

#[cfg(test)]
mod tests;

/// Feature names exactly as recorded from the tiny model (Task 1
/// introspection, pinned by `tests/model_io.rs`); they match the generated
/// Swift wrappers (`Models.swift:909-1107`).
mod names {
  pub const AUDIO: &str = "audio";
  pub const MEL: &str = "melspectrogram_features";
  pub const ENCODER: &str = "encoder_output_embeds";
  pub const INPUT_IDS: &str = "input_ids";
  pub const CACHE_LENGTH: &str = "cache_length";
  pub const KEY_CACHE: &str = "key_cache";
  pub const VALUE_CACHE: &str = "value_cache";
  pub const KV_UPDATE_MASK: &str = "kv_cache_update_mask";
  pub const PADDING_MASK: &str = "decoder_key_padding_mask";
  pub const LOGITS: &str = "logits";
  pub const KEY_UPDATES: &str = "key_cache_updates";
  pub const VALUE_UPDATES: &str = "value_cache_updates";
  pub const ALIGNMENT: &str = "alignment_heads_weights";
}

/// Swift's initial `decoderKeyPaddingMask` fill value
/// (`TextDecoder.swift:143`): additive attention mask, `-10000` hides a KV
/// slot, `0` exposes it.
const PADDING_MASK_HIDDEN: f32 = -10000.0;

/// Dimension `position` of the input feature named `feature`, or
/// [`BackendError::MissingFeature`]. Ports
/// `ModelUtilities.getModelInputDimension`
/// (`ArgmaxCore/ModelUtilities.swift:13-19`); a feature that is present
/// but whose constrained shape lacks `position` is reported as missing
/// too — the dimension this port needs isn't there (Swift would trap on
/// `shape[position]` instead).
fn input_dim(
  model: &Model,
  model_name: &'static str,
  feature: &'static str,
  position: usize,
) -> Result<usize, BackendError> {
  model
    .description()
    .input(feature)
    .and_then(|f| f.shape().get(position).copied())
    .ok_or(BackendError::MissingFeature {
      model: model_name,
      name: feature,
    })
}

/// Output-side twin of [`input_dim`] (`ModelUtilities.swift:22-28`).
fn output_dim(
  model: &Model,
  model_name: &'static str,
  feature: &'static str,
  position: usize,
) -> Result<usize, BackendError> {
  model
    .description()
    .output(feature)
    .and_then(|f| f.shape().get(position).copied())
    .ok_or(BackendError::MissingFeature {
      model: model_name,
      name: feature,
    })
}

// ---------------------------------------------------------------------
// CoreMlDecoderState
// ---------------------------------------------------------------------

/// Pre-allocated, reusable decoder tensors — the port of Swift
/// `DecodingInputs` (`Models.swift:291-323`; allocation
/// `TextDecoder.swift:137-147`). One instance serves a whole transcription:
/// [`CoreMlBackend::decode_step`] mutates it in place and
/// [`CoreMlBackend::reset_decoder_state`] restores the fresh-window
/// invariant between windows.
///
/// **Documented deviation — f32 alignment accumulator:** Swift accumulates
/// alignment weights in an f16 `MLMultiArray` (`alignmentWeights`,
/// `TextDecoder.swift:141`). Here the accumulator is a plain
/// `Vec<f32>` (`max_token_context * n_audio_ctx`, row-major): DTW consumes
/// f32 ([`AlignmentView`] is f32), and the buffer is never a model input,
/// so nothing requires the CoreML tensor type or f16 storage.
///
/// The three scratch `Vec<f16>` buffers are sized once at construction so
/// the per-step output extraction (`copy_into` gathers, see the module
/// doc) performs no whisperkit-level heap allocation per step (spec
/// §10); the remaining per-step allocations are small `Vec<usize>`
/// shape/stride collections inside `coremlit`'s accessors.
#[derive(Debug)]
pub struct CoreMlDecoderState {
  /// `[1] i32` — current token (`TextDecoder.swift:137`).
  input_ids: MultiArray,
  /// `[1] i32` — current KV position (`TextDecoder.swift:138`).
  cache_length: MultiArray,
  /// `[1, kv_dim, 1, max_token_context] f16`, zeroed (`:139`).
  key_cache: MultiArray,
  /// `[1, kv_dim, 1, max_token_context] f16`, zeroed (`:140`).
  value_cache: MultiArray,
  /// `[1, max_token_context]` in the model's declared dtype (`f16` on the
  /// introspected tiny model, though Swift allocates i32 — `:142`);
  /// `[0, 0] = 1`, rest `0` (`:146`).
  kv_cache_update_mask: MultiArray,
  /// `[1, max_token_context] f16`; `[0, 0] = 0`, rest `-10000`
  /// (`:143`, `:147`).
  decoder_key_padding_mask: MultiArray,
  /// f32 alignment accumulator (see the struct doc), row `position + 1`
  /// per decoded token; Swift's `alignmentWeights` (`:141`).
  alignment: Vec<f32>,
  /// Rows of `alignment` written so far (the [`AlignmentView`] row count).
  alignment_rows: usize,
  /// Reused per-step gather target for the `[1, kv_dim, 1, 1]` KV updates.
  kv_scratch: Vec<f16>,
  /// Reused per-step gather target for the `[1, 1, vocab]` logits.
  logits_scratch: Vec<f16>,
  /// Reused per-step gather target for the `[1, n_audio_ctx]` alignment
  /// slice.
  align_scratch: Vec<f16>,
}

/// Ports the decode-loop slice of `updateKVCache`
/// (`TextDecoder.swift:218-270`, slice shape `[1, kv_dim, 1, 1]`): gathers
/// `update` into `scratch` (`copy_into`, since real outputs may be
/// row-padded), then writes `tensor[0, j, 0, position] = slice[0, j, 0, 0]`
/// for every channel `j`. Our caches are `zeros`-allocated and therefore
/// contiguous with strides `[kv_dim * max_ctx, max_ctx, max_ctx, 1]`, so
/// the destination offset is `j * max_ctx + position`.
fn append_kv(
  cache: &mut MultiArray,
  update: &MultiArray,
  scratch: &mut Vec<f16>,
  kv_dim: usize,
  max_ctx: usize,
  position: usize,
) -> Result<(), BackendError> {
  scratch.resize(kv_dim, f16::ZERO);
  update.copy_into::<f16>(scratch)?;
  let dst = cache.as_slice_mut::<f16>()?;
  for (j, &value) in scratch.iter().enumerate() {
    // tensor[0, j, 0, position] = slice[0, j, 0, 0]  (TextDecoder.swift:250-263)
    dst[j * max_ctx + position] = value;
  }
  Ok(())
}

// ---------------------------------------------------------------------
// CoreMlBackend
// ---------------------------------------------------------------------

/// The real [`InferenceBackend`]: owns the three `coremlit::Model`s of a
/// Whisper pipeline and drives them per the tiny model's introspected I/O
/// contract (see the module doc). Construction derives [`ModelDims`] from
/// the models' own descriptions, so non-tiny variants report their real
/// dimensions without any hardcoded table.
#[derive(Debug)]
pub struct CoreMlBackend {
  mel: Model,
  encoder: Model,
  decoder: Model,
  dims: ModelDims,
  supports_alignment: bool,
}

impl CoreMlBackend {
  /// Builds a backend from the three loaded models, deriving every
  /// [`ModelDims`] field from their descriptions (ports
  /// `TextDecoder.swift:309-331` and `FeatureExtractor.swift:25-39`):
  /// `window_samples` from the mel `audio` input's dim 0, `n_mels` from the
  /// mel output's dim 1, `embed_dim`/`n_audio_ctx` from the decoder's
  /// `encoder_output_embeds` input dims 1/3, `kv_dim`/`max_token_context`
  /// from the decoder's `key_cache` input dims 1/3, and `vocab` as the
  /// decoder `logits` output's shape *product* (layout-agnostic: Task 1
  /// recorded `[1, 1, 51865]` where the generated Swift wrapper doc claims
  /// `[1, 51865, 1, 1]`). `supports_word_timestamps` probes the
  /// `alignment_heads_weights` output (`TextDecoder.swift:309-311`).
  ///
  /// # Errors
  /// [`BackendError::MissingFeature`] if any dimension-bearing feature is
  /// absent from its model's description (or its constrained shape lacks
  /// the required dimension).
  pub fn new(mel: Model, encoder: Model, decoder: Model) -> Result<Self, BackendError> {
    let window_samples = input_dim(&mel, "mel", names::AUDIO, 0)?;
    let n_mels = output_dim(&mel, "mel", names::MEL, 1)?;
    let embed_dim = input_dim(&decoder, "decoder", names::ENCODER, 1)?;
    let n_audio_ctx = input_dim(&decoder, "decoder", names::ENCODER, 3)?;
    let kv_dim = input_dim(&decoder, "decoder", names::KEY_CACHE, 1)?;
    let max_token_context = input_dim(&decoder, "decoder", names::KEY_CACHE, 3)?;
    let vocab = match decoder.description().output(names::LOGITS) {
      Some(logits) if !logits.shape().is_empty() => logits.shape().iter().product(),
      _ => {
        return Err(BackendError::MissingFeature {
          model: "decoder",
          name: names::LOGITS,
        });
      }
    };
    // Swift's supportsWordTimestamps is getModelOutputDimension(...) !=
    // nil (TextDecoder.swift:309-311) — the OUTER dim must exist, not just
    // the output; output_dim is the faithful probe.
    let supports_alignment = output_dim(&decoder, "decoder", names::ALIGNMENT, 0).is_ok();

    let dims = ModelDims::new()
      .with_window_samples(window_samples)
      .with_n_mels(n_mels)
      .with_embed_dim(embed_dim)
      .with_n_audio_ctx(n_audio_ctx)
      .with_kv_dim(kv_dim)
      .with_max_token_context(max_token_context)
      .with_vocab(vocab);

    Ok(Self {
      mel,
      encoder,
      decoder,
      dims,
      supports_alignment,
    })
  }

  /// Builds a backend from an already-loaded [`LoadedModels`] triple — the
  /// `ModelManager`-driven construction path (`model::manager`) —
  /// delegating to [`Self::new`] via [`LoadedModels::into_parts`].
  ///
  /// # Errors
  /// As [`Self::new`].
  pub fn from_loaded(models: LoadedModels) -> Result<Self, BackendError> {
    let (mel, encoder, decoder) = models.into_parts();
    Self::new(mel, encoder, decoder)
  }

  /// Whether the decoder carries the cross-attention word-timestamp head
  /// (`alignment_heads_weights`) — Swift `supportsWordTimestamps`
  /// (`TextDecoder.swift:309-311`).
  pub const fn supports_word_timestamps(&self) -> bool {
    self.supports_alignment
  }
}

/// **Documented deviation — KV/mask updates live inside `decode_step`:**
/// Swift's decode loop updates the KV cache, both masks, and the alignment
/// weights *in the loop body*, and skips all of them when the
/// completion-check breaks (`TextDecoder.swift:673-720`). The
/// [`InferenceBackend`] trait keeps decoder tensors opaque, so this port
/// performs the same updates *inside* [`InferenceBackend::decode_step`],
/// unconditionally. Equivalent because (i) after the completion break the
/// loop never issues another step against the same state before a reset,
/// so the extra advance is never observed by a prediction; (ii) the
/// loop keeps positions `<= max_token_context - 2` (`loop_count <=
/// MAX_TOKEN_CONTEXT - 1`), exactly where Swift's conditional updates
/// run — and at the trait-legal last slot, which Swift never reaches,
/// the next-step mask preparation is skipped (nothing to prepare) while
/// the KV/alignment writes still land in their headroom; and (iii)
/// [`InferenceBackend::reset_decoder_state`] restores the full mask/
/// cache-visibility invariant, so the next window starts from the same
/// state either way.
impl InferenceBackend for CoreMlBackend {
  type Features = MultiArray;
  type EncoderOutput = MultiArray;
  type DecoderState = CoreMlDecoderState;

  fn extract_features(&self, audio: &[f32]) -> Result<Self::Features, BackendError> {
    let expected = self.dims.window_samples();
    if audio.len() != expected {
      return Err(BackendError::AudioLength {
        got: audio.len(),
        expected,
      });
    }
    let array = MultiArray::from_slice(&[expected], audio)?;
    let mut outputs = self.mel.predict_with(&[(names::AUDIO, &array)])?;
    outputs
      .take(names::MEL)
      .ok_or(BackendError::MissingFeature {
        model: "mel",
        name: names::MEL,
      })
  }

  fn encode(&self, features: &Self::Features) -> Result<Self::EncoderOutput, BackendError> {
    let mut outputs = self.encoder.predict_with(&[(names::MEL, features)])?;
    outputs
      .take(names::ENCODER)
      .ok_or(BackendError::MissingFeature {
        model: "encoder",
        name: names::ENCODER,
      })
  }

  fn new_decoder_state(&self) -> Result<Self::DecoderState, BackendError> {
    let kv_dim = self.dims.kv_dim();
    let max_ctx = self.dims.max_token_context();

    // TextDecoder.swift:137-143 — zeros() covers Swift's initialValue 0
    // for input_ids/cache_length/key_cache/value_cache.
    let input_ids = MultiArray::zeros(&[1], DataType::I32)?;
    let cache_length = MultiArray::zeros(&[1], DataType::I32)?;
    let key_cache = MultiArray::zeros(&[1, kv_dim, 1, max_ctx], DataType::F16)?;
    let value_cache = MultiArray::zeros(&[1, kv_dim, 1, max_ctx], DataType::F16)?;

    // The update mask's dtype is the live description's truth — never a
    // guess: Swift allocates i32 (TextDecoder.swift:142) but the compiled
    // tiny model declares f16 (Task 1, pinned by tests/model_io.rs), and
    // CoreML rejects mistyped inputs at predict time. The f16 fallback is
    // that same recorded truth, used only if the description leaves the
    // dtype unconstrained. Should a model generation genuinely declare
    // i32 here, the f16 mask writes below fail loudly with a structured
    // dtype-mismatch error at construction rather than corrupting a mask.
    let update_mask_dtype = self
      .decoder
      .description()
      .input(names::KV_UPDATE_MASK)
      .and_then(|f| f.data_type())
      .unwrap_or(DataType::F16);
    let mut kv_cache_update_mask = MultiArray::zeros(&[1, max_ctx], update_mask_dtype)?;
    let mut decoder_key_padding_mask = MultiArray::zeros(&[1, max_ctx], DataType::F16)?;

    // TextDecoder.swift:143 + :146-147 — every slot hidden except slot 0,
    // which is this window's first update target.
    decoder_key_padding_mask
      .as_slice_mut::<f16>()?
      .fill(f16::from_f32(PADDING_MASK_HIDDEN));
    decoder_key_padding_mask.fill_at(&[0, 0], f16::ZERO)?;
    kv_cache_update_mask.fill_at(&[0, 0], f16::ONE)?;

    #[cfg(debug_assertions)]
    for (name, array) in [
      (names::INPUT_IDS, &input_ids),
      (names::CACHE_LENGTH, &cache_length),
      (names::KEY_CACHE, &key_cache),
      (names::VALUE_CACHE, &value_cache),
      (names::KV_UPDATE_MASK, &kv_cache_update_mask),
      (names::PADDING_MASK, &decoder_key_padding_mask),
    ] {
      if let Some(feature) = self.decoder.description().input(name) {
        debug_assert_eq!(feature.shape(), array.shape(), "{name} shape");
        debug_assert_eq!(feature.data_type(), Some(array.data_type()), "{name} dtype");
      }
    }

    Ok(CoreMlDecoderState {
      input_ids,
      cache_length,
      key_cache,
      value_cache,
      kv_cache_update_mask,
      decoder_key_padding_mask,
      // One row of headroom, exactly like MockBackend: a step at the
      // trait-legal last position (`max_ctx - 1`) writes alignment row
      // `position + 1 == max_ctx`, so the buffer holds `max_ctx + 1` rows.
      alignment: vec![0.0; (max_ctx + 1) * self.dims.n_audio_ctx()],
      alignment_rows: 0,
      // Sized up front so even the first decode step allocates nothing.
      kv_scratch: vec![f16::ZERO; kv_dim],
      logits_scratch: vec![f16::ZERO; self.dims.vocab()],
      align_scratch: vec![f16::ZERO; self.dims.n_audio_ctx()],
    })
  }

  fn reset_decoder_state(&self, state: &mut Self::DecoderState) {
    // Ports DecodingInputs.reset (Models.swift:312-322): cache_length back
    // to 0 and both masks back to the fresh-window state. As in Swift,
    // input_ids (overwritten every step) and the KV caches (dead data
    // beyond cache_length, masked off by the padding mask) are left as-is.
    // The expects are on this state's own self-allocated arrays — always
    // contiguous, always the written dtype — so they cannot fire for any
    // state produced by `new_decoder_state`.
    state
      .cache_length
      .fill_at(&[0], 0_i32)
      .expect("cache_length is a self-allocated contiguous [1] i32 array");
    let padding = state
      .decoder_key_padding_mask
      .as_slice_mut::<f16>()
      .expect("padding mask is a self-allocated contiguous f16 array");
    padding.fill(f16::from_f32(PADDING_MASK_HIDDEN));
    padding[0] = f16::ZERO;
    let update = state
      .kv_cache_update_mask
      .as_slice_mut::<f16>()
      .expect("update mask is a self-allocated contiguous f16 array");
    update.fill(f16::ZERO);
    update[0] = f16::ONE;
    // Swift leaves the weights tensor itself; our view exposes
    // `alignment_rows`, and zeroing keeps a view honest even at row
    // granularity after the row count resets.
    state.alignment.fill(0.0);
    state.alignment_rows = 0;
  }

  fn decode_step(
    &self,
    token: u32,
    position: usize,
    encoder_output: &Self::EncoderOutput,
    state: &mut Self::DecoderState,
    logits: &mut Vec<f32>,
  ) -> Result<(), BackendError> {
    let max_ctx = self.dims.max_token_context();
    // The KV slot must exist (trait contract: position in
    // 0..max_token_context). Checked up front with the same structured
    // error a strided write would report, because `append_kv` below
    // indexes a raw slice.
    if position >= max_ctx {
      return Err(BackendError::Tensor(TensorError::IndexOutOfBounds {
        index: position,
        len: max_ctx,
      }));
    }

    // TextDecoder.swift:600-602.
    state.input_ids.fill_at(&[0], token as i32)?;
    state.cache_length.fill_at(&[0], position as i32)?;

    // The seven decoder inputs (TextDecoderMLMultiArrayInputType,
    // TextDecoder.swift:617-625): six state-owned arrays plus the borrowed
    // encoder output — no per-step tensor construction.
    let mut outputs = self.decoder.predict_with(&[
      (names::INPUT_IDS, &state.input_ids),
      (names::CACHE_LENGTH, &state.cache_length),
      (names::KEY_CACHE, &state.key_cache),
      (names::VALUE_CACHE, &state.value_cache),
      (names::KV_UPDATE_MASK, &state.kv_cache_update_mask),
      (names::ENCODER, encoder_output),
      (names::PADDING_MASK, &state.decoder_key_padding_mask),
    ])?;

    // Logits: gather f16 (possibly row-padded — module doc) into scratch,
    // then fully overwrite the caller's buffer with one f32 conversion
    // pass, leaving it exactly vocab() long per the trait contract.
    let logits_array = outputs
      .take(names::LOGITS)
      .ok_or(BackendError::MissingFeature {
        model: "decoder",
        name: names::LOGITS,
      })?;
    state.logits_scratch.resize(self.dims.vocab(), f16::ZERO);
    logits_array.copy_into::<f16>(&mut state.logits_scratch)?;
    logits.clear();
    logits.extend(state.logits_scratch.iter().map(|v| v.to_f32()));

    // KV append (updateKVCache, TextDecoder.swift:218-270 via :688-702).
    let key_updates = outputs
      .take(names::KEY_UPDATES)
      .ok_or(BackendError::MissingFeature {
        model: "decoder",
        name: names::KEY_UPDATES,
      })?;
    let value_updates = outputs
      .take(names::VALUE_UPDATES)
      .ok_or(BackendError::MissingFeature {
        model: "decoder",
        name: names::VALUE_UPDATES,
      })?;
    let kv_dim = self.dims.kv_dim();
    append_kv(
      &mut state.key_cache,
      &key_updates,
      &mut state.kv_scratch,
      kv_dim,
      max_ctx,
      position,
    )?;
    append_kv(
      &mut state.value_cache,
      &value_updates,
      &mut state.kv_scratch,
      kv_dim,
      max_ctx,
      position,
    )?;

    // Mask flips (TextDecoder.swift:704-707), in the mask's introspected
    // dtype: expose the next slot, and move the update target from this
    // position to the next. Their only purpose is preparing the NEXT
    // step, so at the trait-legal last slot (position == max_ctx - 1,
    // which Swift's own loop bound never reaches) there is nothing to
    // prepare and all three writes are skipped as a unit — the state
    // stays internally consistent and, as always, only a reset makes it
    // steppable again.
    if position + 1 < max_ctx {
      state
        .decoder_key_padding_mask
        .fill_at(&[0, position + 1], f16::ZERO)?;
      state
        .kv_cache_update_mask
        .fill_at(&[0, position], f16::ZERO)?;
      state
        .kv_cache_update_mask
        .fill_at(&[0, position + 1], f16::ONE)?;
    }

    // Alignment (updateAlignmentWeights, TextDecoder.swift:272-296 via
    // :709-717): row `position + 1`, converted to the f32 accumulator
    // (deviation documented on `CoreMlDecoderState`). Presence-gated per
    // step exactly like Swift's `if let ... = cache?.alignmentWeights`.
    if self.supports_alignment
      && let Some(alignment) = outputs.take(names::ALIGNMENT)
    {
      let cols = self.dims.n_audio_ctx();
      state.align_scratch.resize(cols, f16::ZERO);
      alignment.copy_into::<f16>(&mut state.align_scratch)?;
      let start = (position + 1) * cols;
      // In bounds: position < max_ctx (checked at entry), so
      // start + cols == (position + 2) * cols <= (max_ctx + 1) * cols
      // == alignment.len() (the buffer's one-row headroom).
      for (dst, src) in state.alignment[start..start + cols]
        .iter_mut()
        .zip(&state.align_scratch)
      {
        *dst = src.to_f32();
      }
      state.alignment_rows = state.alignment_rows.max(position + 2);
    }

    Ok(())
  }

  fn alignment_weights<'state>(
    &self,
    state: &'state Self::DecoderState,
  ) -> Option<AlignmentView<'state>> {
    self.supports_alignment.then(|| {
      let cols = self.dims.n_audio_ctx();
      AlignmentView::new(
        &state.alignment[..state.alignment_rows * cols],
        state.alignment_rows,
        cols,
      )
    })
  }

  fn dims(&self) -> ModelDims {
    self.dims
  }
}
