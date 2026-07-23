//! The siglip [`ImageEmbedder`]: the NaFlex host-side preprocessing (the private
//! `preprocess` submodule) around the fp16 CoreML vision graph, with L2
//! normalization applied in Rust.

mod preprocess;

use core::fmt;
use std::path::Path;

use crate::{ComputeUnits, DataType, Model, ModelDescription, MultiArray};

use crate::embeddings::siglip::{
  embedding::{EMBEDDING_DIM, Embedding, check_finite_output},
  error::{Error, Result},
  image::preprocess::{parse_base_pos_grid, preprocess_image},
};

pub use preprocess::PATCH_DIM;

/// Declared feature names on the siglip vision `.mlmodelc` (pinned by
/// `tests/siglip/model_io.rs`).
mod names {
  pub const PIXEL_VALUES: &str = "pixel_values";
  pub const POSITION_EMBEDDINGS: &str = "position_embeddings";
  pub const ATTENTION_MASK: &str = "attention_mask";
  pub const IMAGE_FEATURES: &str = "image_features";
}

/// Default [`ImageEmbedderOptions::compute`]: [`ComputeUnits::CpuAndGpu`] — the
/// **measured floor-holding** placement, deliberately NOT [`ComputeUnits::All`].
///
/// The conversion probe measured the vision graph as **99.1% ANE-preferred** in
/// its CoreML compute plan, yet its fp16-on-ANE parity is **0.998118 worst**,
/// *below* the committed **0.99917** floor; the identical fp16 weights reach
/// **0.999959** on the GPU (fp32 accumulation). Because the planner prefers the
/// ANE for nearly every op, [`ComputeUnits::All`] risks silently dispatching the
/// vision tower to the below-floor ANE arm. `CpuAndGpu` pins the graph to the
/// floor-holding GPU path (mirroring `clap`'s measure-then-pin `text` default).
///
/// This is a deliberate spec deviation pending the conversion runbook's `All`-arm
/// characterization (placement + parity + compute-plan dispatch); if `All`
/// measures GPU-identical and floor-holding, the default may move to `All` in a
/// follow-up — with the measurement in hand. Every unit stays selectable via
/// [`ImageEmbedderOptions::with_compute`] / [`ImageEmbedderOptions::set_compute`];
/// placement is characterized, not asserted (`tests/siglip/placement.rs`).
pub const DEFAULT_IMAGE_COMPUTE: ComputeUnits = ComputeUnits::CpuAndGpu;

#[cfg(feature = "serde")]
fn default_image_compute() -> ComputeUnits {
  DEFAULT_IMAGE_COMPUTE
}

/// Construction options for [`ImageEmbedder`] (rust-options-pattern): a single
/// `compute` knob with one source of truth shared by `const new`/`Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ImageEmbedderOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_image_compute",
      with = "crate::embeddings::siglip::compute_units_serde"
    )
  )]
  compute: ComputeUnits,
}

impl Default for ImageEmbedderOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl ImageEmbedderOptions {
  /// Options matching the module default: [`DEFAULT_IMAGE_COMPUTE`].
  pub const fn new() -> Self {
    Self {
      compute: DEFAULT_IMAGE_COMPUTE,
    }
  }

  /// Which hardware CoreML may schedule the vision graph on.
  #[inline]
  pub const fn compute(&self) -> ComputeUnits {
    self.compute
  }

  /// Builder form of [`Self::set_compute`].
  #[must_use]
  #[inline]
  pub const fn with_compute(mut self, compute: ComputeUnits) -> Self {
    self.set_compute(compute);
    self
  }

  /// Sets [`Self::compute`] in place.
  #[inline]
  pub const fn set_compute(&mut self, compute: ComputeUnits) -> &mut Self {
    self.compute = compute;
    self
  }
}

/// A borrowed view of one decoded RGB8 image: a `width · height · 3` row-major,
/// RGB-interleaved `&[u8]` (the sans-I/O seam — decoding PNG/JPEG is the
/// caller's responsibility, like `clap`'s 48 kHz resampling, so the crate gains
/// no image-decoder runtime dependency).
///
/// [`Rgb8Image::new`] validates the geometry so preprocessing can index without
/// bounds surprises.
#[derive(Debug, Clone, Copy)]
pub struct Rgb8Image<'a> {
  data: &'a [u8],
  width: usize,
  height: usize,
}

impl<'a> Rgb8Image<'a> {
  /// Wrap a decoded RGB8 buffer, validating its geometry.
  ///
  /// `data` must be exactly `width · height · 3` bytes, row-major with the three
  /// RGB channels interleaved per pixel.
  ///
  /// # Errors
  /// [`Error::ImageDimensions`] if `width` or `height` is zero, or if
  /// `width · height · 3` overflows `usize`; [`Error::ImageDataLength`] if
  /// `data.len()` is not exactly `width · height · 3`.
  pub fn new(data: &'a [u8], width: usize, height: usize) -> Result<Self> {
    if width == 0 || height == 0 {
      return Err(Error::ImageDimensions { width, height });
    }
    let expected = width
      .checked_mul(height)
      .and_then(|hw| hw.checked_mul(3))
      .ok_or(Error::ImageDimensions { width, height })?;
    if data.len() != expected {
      return Err(Error::ImageDataLength {
        got: data.len(),
        expected,
      });
    }
    Ok(Self {
      data,
      width,
      height,
    })
  }

  /// The image width in pixels.
  #[inline]
  pub const fn width(&self) -> usize {
    self.width
  }

  /// The image height in pixels.
  #[inline]
  pub const fn height(&self) -> usize {
    self.height
  }

  /// The backing RGB8 bytes (`width · height · 3`, row-major, interleaved).
  #[inline]
  pub const fn data(&self) -> &'a [u8] {
    self.data
  }
}

/// Caller-supplied NaFlex-preprocessed vision tensors for
/// [`ImageEmbedder::embed_preprocessed`]: the graph's three inputs —
/// `pixel_values` `[1, P, `[`PATCH_DIM`]`]`, `position_embeddings`
/// `[1, P, `[`EMBEDDING_DIM`]`]`, `attention_mask` `[1, P]` — flattened
/// row-major and **already padded to the patch budget** `P`
/// ([`ImageEmbedder::max_num_patches`]), exactly as the NaFlex pipeline emits
/// them: real patch rows as a contiguous prefix (mask `1.0`), zero-filled pad
/// rows after (mask `0.0`).
///
/// [`PreprocessedImage::try_new`] validates shape and structure once; the
/// tensors are immutable afterwards (private fields, no mutators), so a
/// constructed value stays valid. [`ImageEmbedder::preprocess`] produces this
/// type from a decoded [`Rgb8Image`] — the in-crate reference for what
/// `try_new` accepts.
///
/// **What validation cannot see:** whether the tensors were produced by the
/// exact NaFlex pipeline this model was converted against — the
/// antialiased-bilinear resize coefficients, the `((x/255) − 0.5)/0.5`
/// normalization, the `(patch_row, patch_col, py, px, channel)` flatten order,
/// and the base-grid position-embedding lift. Tensors from a deviating
/// pipeline pass validation and **silently degrade** the embedding — no error
/// is raised. Unless inputs must be precomputed offline, use
/// [`ImageEmbedder::embed`], the safe default.
#[derive(Clone)]
pub struct PreprocessedImage {
  /// `[max_num_patches · PATCH_DIM]`: real patch rows, then zero pad rows.
  pixel_values: Vec<f32>,
  /// `[max_num_patches · EMBEDDING_DIM]`: lifted rows, then zero pad rows.
  position_embeddings: Vec<f32>,
  /// `[max_num_patches]`: exactly `1.0` on the real-patch prefix, `0.0` after.
  attention_mask: Vec<f32>,
  /// The patch budget `P` the lengths were validated against.
  max_num_patches: usize,
}

impl PreprocessedImage {
  /// Validates and wraps a padded NaFlex tensor bundle for a model whose
  /// resolved patch budget is `max_num_patches`
  /// ([`ImageEmbedder::max_num_patches`]).
  ///
  /// Checks, in order: the budget is usable (non-zero, lengths representable);
  /// exact lengths (`pixel_values` = `max_num_patches · `[`PATCH_DIM`],
  /// `position_embeddings` = `max_num_patches · `[`EMBEDDING_DIM`],
  /// `attention_mask` = `max_num_patches`); `pixel_values` and
  /// `position_embeddings` are finite; the mask is an exact binary
  /// prefix mask (every entry exactly `0.0` or `1.0`, all `1.0`s before the
  /// first `0.0`, at least one `1.0` — the mask's domain check subsumes its
  /// finiteness); and every padded (mask `0.0`) row of `pixel_values` and
  /// `position_embeddings` is all-zero, as the NaFlex pipeline emits and as
  /// the module's parity evidence covers (fail-closed).
  ///
  /// It **cannot** validate that the values came from the exact NaFlex
  /// pipeline (see the type docs): that is the caller's contract.
  ///
  /// # Errors
  /// [`Error::PreprocessedPatchBudget`] if `max_num_patches` is zero or too
  /// large for the tensor lengths to be representable;
  /// [`Error::PreprocessedLength`] on a wrong tensor length;
  /// [`Error::PreprocessedNonFinite`] on a NaN/infinite `pixel_values` or
  /// `position_embeddings` element; [`Error::PreprocessedMaskValue`] /
  /// [`Error::PreprocessedMaskOrder`] / [`Error::PreprocessedMaskEmpty`] on a
  /// non-binary, non-prefix, or all-pad mask;
  /// [`Error::PreprocessedPadNonZero`] on a nonzero value inside a padded row.
  pub fn try_new(
    pixel_values: Vec<f32>,
    position_embeddings: Vec<f32>,
    attention_mask: Vec<f32>,
    max_num_patches: usize,
  ) -> Result<Self> {
    validate_budget_and_lengths(
      &pixel_values,
      &position_embeddings,
      &attention_mask,
      max_num_patches,
    )?;
    check_tensor_finite(names::PIXEL_VALUES, &pixel_values)?;
    check_tensor_finite(names::POSITION_EMBEDDINGS, &position_embeddings)?;
    let num_real = validate_mask(&attention_mask)?;
    validate_pad_rows(names::PIXEL_VALUES, &pixel_values, num_real, PATCH_DIM)?;
    validate_pad_rows(
      names::POSITION_EMBEDDINGS,
      &position_embeddings,
      num_real,
      EMBEDDING_DIM,
    )?;
    Ok(Self {
      pixel_values,
      position_embeddings,
      attention_mask,
      max_num_patches,
    })
  }

  /// Module-internal constructor for the pipeline's own outputs, whose
  /// structural invariants (lengths, binary prefix mask, zero pads) hold by
  /// construction in `patchify` / `lift_position_embeddings` (cf.
  /// `Embedding::from_array_trusted_unit_norm`). Deliberately does NOT assert
  /// finiteness: a non-finite value in a caller-supplied position-grid sidecar
  /// must keep flowing to the same typed predict-time error it always did, not
  /// become a debug panic.
  fn from_pipeline(
    pixel_values: Vec<f32>,
    position_embeddings: Vec<f32>,
    attention_mask: Vec<f32>,
    max_num_patches: usize,
  ) -> Self {
    debug_assert!(
      validate_structural(
        &pixel_values,
        &position_embeddings,
        &attention_mask,
        max_num_patches
      )
      .is_ok(),
      "internal NaFlex pipeline emitted a structurally invalid tensor bundle"
    );
    Self {
      pixel_values,
      position_embeddings,
      attention_mask,
      max_num_patches,
    }
  }

  /// The patch budget `P` this bundle was validated against — must equal the
  /// target embedder's [`ImageEmbedder::max_num_patches`].
  #[inline]
  pub const fn max_num_patches(&self) -> usize {
    self.max_num_patches
  }

  /// The flattened `[max_num_patches · `[`PATCH_DIM`]`]` `pixel_values` rows.
  #[inline]
  pub fn pixel_values(&self) -> &[f32] {
    &self.pixel_values
  }

  /// The flattened `[max_num_patches · `[`EMBEDDING_DIM`]`]`
  /// `position_embeddings` rows.
  #[inline]
  pub fn position_embeddings(&self) -> &[f32] {
    &self.position_embeddings
  }

  /// The `[max_num_patches]` real/pad `attention_mask` (`1.0` real prefix,
  /// `0.0` pads).
  #[inline]
  pub fn attention_mask(&self) -> &[f32] {
    &self.attention_mask
  }
}

impl fmt::Debug for PreprocessedImage {
  /// Compact — elides the megabyte-scale tensors (the `Embedding` Debug
  /// convention), showing the budget and the mask's real-prefix length.
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let num_real = self
      .attention_mask
      .iter()
      .take_while(|&&v| v == 1.0)
      .count();
    f.debug_struct("PreprocessedImage")
      .field("max_num_patches", &self.max_num_patches)
      .field("num_real_patches", &num_real)
      .finish_non_exhaustive()
  }
}

/// siglip vision embedder: a decoded [`Rgb8Image`] in, a unit-norm 768-d
/// [`Embedding`] out — the same joint-space [`Embedding`] the text tower emits.
///
/// The front-end is a Rust NaFlex port (the private `preprocess` submodule):
/// it fits the image to the resolved patch budget `P`, resizes with a uint8
/// PIL-parity antialiased-bilinear kernel, normalizes, patchifies into the
/// graph's `[1, P, 768]` `pixel_values` + `[1, P]` `attention_mask`, and lifts
/// the base position grid into `[1, P, 768]` `position_embeddings`. The fp16
/// CoreML graph maps those to a pre-normalization 768-d projection, which this
/// embedder L2-normalizes.
///
/// `&self` inference: preprocessing scratch is per-call local, so fan-out means
/// one [`ImageEmbedder`] per worker over a `Send` (but deliberately `!Sync`)
/// [`crate::Model`].
#[derive(Debug)]
pub struct ImageEmbedder {
  model: Model,
  /// The base `16×16×768` position grid (parsed from the `.f32le.bin` sidecar),
  /// resized per image by the pos-emb lift.
  base_pos_embed: Vec<f32>,
  /// The patch budget `P` resolved from the loaded model's `pixel_values [1, P,
  /// 768]` contract (D2 — never a code constant; a 256/1024 tier is a drop-in
  /// artifact with no inference-core change).
  max_num_patches: usize,
}

impl ImageEmbedder {
  /// Loads the vision `.mlmodelc` and its base position-grid sidecar
  /// (`pos_embed_16x16x768.f32le.bin`) from paths, with custom `options` — the
  /// primary constructor. Resolves the patch budget `P` and validates the I/O
  /// contract against the metadata at load.
  ///
  /// # Errors
  /// [`Error::PosEmbedLoad`] if the sidecar is unreadable; [`Error::PosEmbedLength`]
  /// if its byte length is not the exact `16·16·768·4` grid; [`Error::Load`] if
  /// CoreML rejects the model; [`Error::ContractMismatch`] if the model's I/O
  /// contract mismatches the vision graph contract.
  pub fn load(
    model_path: impl AsRef<Path>,
    pos_embed_path: impl AsRef<Path>,
    options: ImageEmbedderOptions,
  ) -> Result<Self> {
    let bytes = std::fs::read(pos_embed_path.as_ref()).map_err(Error::PosEmbedLoad)?;
    let base_pos_embed = parse_base_pos_grid(&bytes)?;
    Self::from_parts(model_path, base_pos_embed, options)
  }

  /// Loads the vision model and sidecar from paths using
  /// [`ImageEmbedderOptions::new`].
  ///
  /// # Errors
  /// As [`Self::load`].
  pub fn from_files(
    model_path: impl AsRef<Path>,
    pos_embed_path: impl AsRef<Path>,
  ) -> Result<Self> {
    Self::load(model_path, pos_embed_path, ImageEmbedderOptions::new())
  }

  /// Loads the vision model from a path and the base position grid from
  /// caller-supplied bytes (raw little-endian f32, exactly `16·16·768·4` bytes).
  ///
  /// # Errors
  /// As [`Self::load`] (minus [`Error::PosEmbedLoad`] — the caller owns the
  /// bytes); [`Error::PosEmbedLength`] on a wrong byte length.
  pub fn from_memory(
    model_path: impl AsRef<Path>,
    pos_embed_bytes: &[u8],
    options: ImageEmbedderOptions,
  ) -> Result<Self> {
    let base_pos_embed = parse_base_pos_grid(pos_embed_bytes)?;
    Self::from_parts(model_path, base_pos_embed, options)
  }

  fn from_parts(
    model_path: impl AsRef<Path>,
    base_pos_embed: Vec<f32>,
    options: ImageEmbedderOptions,
  ) -> Result<Self> {
    let model = Model::load(model_path, options.compute())?;
    let max_num_patches = resolve_patch_budget(model.description())?;
    Ok(Self {
      model,
      base_pos_embed,
      max_num_patches,
    })
  }

  /// The patch budget `P` this model was converted at — resolved from the
  /// loaded `pixel_values [1, P, 768]` contract (D2), not a code constant.
  #[inline]
  pub const fn max_num_patches(&self) -> usize {
    self.max_num_patches
  }

  /// Runs the module's own NaFlex pipeline on `image` without predicting:
  /// pure host-side math (no CoreML call), producing the validated
  /// [`PreprocessedImage`] that [`Self::embed_preprocessed`] accepts.
  /// `embed(image)` is exactly `embed_preprocessed(&preprocess(image)?)`;
  /// capture the bundle to embed later, or to feed another embedder of the
  /// same patch budget.
  ///
  /// # Errors
  /// [`Error::PatchCount`] if preprocessing overflows the budget (a solver
  /// bug — the defensive backstop, as in [`Self::embed`]).
  pub fn preprocess(&self, image: Rgb8Image<'_>) -> Result<PreprocessedImage> {
    let inputs = preprocess_image(
      image.data(),
      image.width(),
      image.height(),
      &self.base_pos_embed,
      self.max_num_patches,
    )?;
    Ok(PreprocessedImage::from_pipeline(
      inputs.pixel_values,
      inputs.position_embeddings,
      inputs.attention_mask,
      self.max_num_patches,
    ))
  }

  /// Embeds one decoded image into a unit-norm [`Embedding`].
  ///
  /// # Errors
  /// [`Error::PatchCount`] if preprocessing overflows the budget (a solver bug);
  /// [`Error::PreprocessAllocation`] if a resize working buffer cannot be sized
  /// or reserved (pathological source geometry);
  /// [`Error::Tensor`] / [`Error::Prediction`] on a tensor or CoreML failure;
  /// [`Error::OutputShape`] if the predicted `image_features` shape diverges from
  /// `[1, `[`EMBEDDING_DIM`]`]`; [`Error::NonFiniteOutput`] if the model output
  /// has a NaN/infinite component — model corruption, classified apart from a
  /// caller's own non-finite embedding data ([`Error::NonFiniteEmbedding`]);
  /// [`Error::EmbeddingZero`] if the (finite) projection has zero magnitude.
  pub fn embed(&self, image: Rgb8Image<'_>) -> Result<Embedding> {
    let inputs = self.preprocess(image)?;
    self.embed_preprocessed(&inputs)
  }

  /// Embeds caller-supplied NaFlex-preprocessed tensors, skipping this
  /// embedder's own preprocessing — the bring-your-own-tensors bypass for
  /// pipelines that run the exact NaFlex front-end offline/batch.
  /// [`Self::embed`] routes through this method, so `embed(image)` ≡
  /// `embed_preprocessed(&preprocess(image)?)` by construction.
  ///
  /// The bundle's shape and structure were validated at
  /// [`PreprocessedImage::try_new`]; here only the patch-budget binding is
  /// checked against this model's resolved `P`
  /// ([`Self::max_num_patches`]). coremlit cannot verify the tensor VALUES
  /// came from the exact NaFlex pipeline this model was converted against —
  /// tensors from a deviating pipeline pass validation and **silently
  /// degrade** the embedding (see [`PreprocessedImage`]). Prefer
  /// [`Self::embed`] unless inputs must be precomputed.
  ///
  /// # Errors
  /// [`Error::PatchBudgetMismatch`] if `inputs` was validated against a
  /// different patch budget than this model resolved at load (e.g. a
  /// 256-tier bundle fed to a 512-tier model); otherwise as
  /// [`Self::embed`]'s predict path: [`Error::Tensor`] /
  /// [`Error::Prediction`] on a tensor or CoreML failure;
  /// [`Error::OutputShape`] if the predicted `image_features` shape diverges
  /// from `[1, `[`EMBEDDING_DIM`]`]`; [`Error::NonFiniteOutput`] on a
  /// NaN/infinite model output; [`Error::EmbeddingZero`] if the (finite)
  /// projection has zero magnitude.
  pub fn embed_preprocessed(&self, inputs: &PreprocessedImage) -> Result<Embedding> {
    check_patch_budget(inputs.max_num_patches(), self.max_num_patches)?;
    self.predict_embedding(
      inputs.pixel_values(),
      inputs.position_embeddings(),
      inputs.attention_mask(),
    )
  }

  /// Shared predict tail of [`Self::embed`] / [`Self::embed_preprocessed`]:
  /// builds the three input tensors, predicts, validates the `image_features`
  /// contract, and L2-normalizes.
  fn predict_embedding(
    &self,
    pixel_values: &[f32],
    position_embeddings: &[f32],
    attention_mask: &[f32],
  ) -> Result<Embedding> {
    let pixel_values = MultiArray::from_slice(&[1, self.max_num_patches, PATCH_DIM], pixel_values)?;
    let position_embeddings = MultiArray::from_slice(
      &[1, self.max_num_patches, EMBEDDING_DIM],
      position_embeddings,
    )?;
    let attention_mask = MultiArray::from_slice(&[1, self.max_num_patches], attention_mask)?;

    let mut outputs = self.model.predict_with(&[
      (names::PIXEL_VALUES, &pixel_values),
      (names::POSITION_EMBEDDINGS, &position_embeddings),
      (names::ATTENTION_MASK, &attention_mask),
    ])?;
    let feats =
      outputs
        .take(names::IMAGE_FEATURES)
        .ok_or_else(|| crate::PredictionError::MissingOutput {
          name: names::IMAGE_FEATURES.to_string(),
        })?;
    if feats.shape() != [1, EMBEDDING_DIM] {
      return Err(Error::OutputShape {
        got: feats.shape().to_vec(),
        expected: vec![1, EMBEDDING_DIM],
      });
    }

    let mut row = [0.0f32; EMBEDDING_DIM];
    feats.copy_into::<f32>(&mut row)?;
    // Classify a NaN/∞ the CoreML runtime produced as model-output corruption
    // (`NonFiniteOutput`) before it reaches `from_slice_normalizing`.
    check_finite_output(&row)?;
    Embedding::from_slice_normalizing(&row)
  }

  /// Runs one throwaway [`Self::embed`] to fully specialize the prediction path,
  /// so the first user-facing request is warm. Construction pays the model load;
  /// this pays the first prediction's graph specialization. Then **reuse** this
  /// same embedder for every request (it is `&self`).
  ///
  /// # Errors
  /// As [`Self::embed`] (the warm-up uses a fixed synthetic image, so no caller
  /// input is read); a failure surfaces a broken model at prewarm time rather
  /// than on the first request.
  pub fn prewarm(&self) -> Result<()> {
    // A fixed 64×64 mid-gray image: valid geometry, non-degenerate, upscaled by
    // NaFlex to the full patch budget exactly as a real image is.
    let data = vec![128u8; 64 * 64 * 3];
    let image = Rgb8Image::new(&data, 64, 64)?;
    self.embed(image)?;
    Ok(())
  }
}

/// Resolves the patch budget `P` from the loaded vision model's `pixel_values
/// [1, P, 768]` contract and validates the full I/O contract against it (D2 +
/// cross-input consistency): `pixel_values` and `position_embeddings` are both
/// `[1, P, 768]` f32, `attention_mask` is `[1, P]` f32, and `image_features` is
/// `[1, 768]` f32 — the same `P` across all three inputs.
fn resolve_patch_budget(description: &ModelDescription) -> Result<usize> {
  // pixel_values [1, P, PATCH_DIM] f32 — the source of the resolved P.
  let pv_expected = format!("[1, P, {PATCH_DIM}] float32");
  let pixel_values =
    description
      .input(names::PIXEL_VALUES)
      .ok_or_else(|| Error::ContractMismatch {
        feature: names::PIXEL_VALUES,
        expected: pv_expected.clone(),
        actual: "missing".to_string(),
      })?;
  let shape = pixel_values.shape();
  if shape.len() != 3
    || shape[0] != 1
    || shape[2] != PATCH_DIM
    || pixel_values.data_type() != Some(DataType::F32)
  {
    return Err(Error::ContractMismatch {
      feature: names::PIXEL_VALUES,
      expected: pv_expected,
      actual: describe(shape, pixel_values.data_type()),
    });
  }
  let p = shape[1];
  if p == 0 {
    return Err(Error::ContractMismatch {
      feature: names::PIXEL_VALUES,
      expected: pv_expected,
      actual: describe(shape, pixel_values.data_type()),
    });
  }

  // position_embeddings [1, P, EMBEDDING_DIM] f32 — same resolved P.
  check_input(
    description,
    names::POSITION_EMBEDDINGS,
    &[1, p, EMBEDDING_DIM],
  )?;
  // attention_mask [1, P] f32 (note: f32, not int32) — same resolved P.
  check_input(description, names::ATTENTION_MASK, &[1, p])?;

  // image_features [1, EMBEDDING_DIM] f32.
  let out_expected = format!("[1, {EMBEDDING_DIM}] float32");
  let output =
    description
      .output(names::IMAGE_FEATURES)
      .ok_or_else(|| Error::ContractMismatch {
        feature: names::IMAGE_FEATURES,
        expected: out_expected.clone(),
        actual: "missing".to_string(),
      })?;
  if output.shape() != [1, EMBEDDING_DIM] || output.data_type() != Some(DataType::F32) {
    return Err(Error::ContractMismatch {
      feature: names::IMAGE_FEATURES,
      expected: out_expected,
      actual: describe(output.shape(), output.data_type()),
    });
  }

  Ok(p)
}

/// Validates one f32 input feature against an exact resolved shape.
fn check_input(
  description: &ModelDescription,
  name: &'static str,
  expected_shape: &[usize],
) -> Result<()> {
  let expected = format!("{expected_shape:?} float32");
  let input = description
    .input(name)
    .ok_or_else(|| Error::ContractMismatch {
      feature: name,
      expected: expected.clone(),
      actual: "missing".to_string(),
    })?;
  if input.shape() != expected_shape || input.data_type() != Some(DataType::F32) {
    return Err(Error::ContractMismatch {
      feature: name,
      expected,
      actual: describe(input.shape(), input.data_type()),
    });
  }
  Ok(())
}

/// Budget + exact-length validation for a preprocessed bundle. The budget
/// guard makes every `max_num_patches · row_dim` product below (and in
/// [`validate_pad_rows`]) provably in-range, so plain multiplication is safe.
fn validate_budget_and_lengths(
  pixel_values: &[f32],
  position_embeddings: &[f32],
  attention_mask: &[f32],
  max_num_patches: usize,
) -> Result<()> {
  // PATCH_DIM and EMBEDDING_DIM coincide (768) but are distinct quantities;
  // guard against the larger so both products stay in range by construction.
  const MAX_ROW_DIM: usize = if PATCH_DIM > EMBEDDING_DIM {
    PATCH_DIM
  } else {
    EMBEDDING_DIM
  };
  if max_num_patches == 0 || max_num_patches > usize::MAX / MAX_ROW_DIM {
    return Err(Error::PreprocessedPatchBudget { max_num_patches });
  }
  check_len(
    names::PIXEL_VALUES,
    pixel_values.len(),
    max_num_patches * PATCH_DIM,
  )?;
  check_len(
    names::POSITION_EMBEDDINGS,
    position_embeddings.len(),
    max_num_patches * EMBEDDING_DIM,
  )?;
  check_len(names::ATTENTION_MASK, attention_mask.len(), max_num_patches)?;
  Ok(())
}

/// One exact-length check, reported as [`Error::PreprocessedLength`].
fn check_len(feature: &'static str, got: usize, expected: usize) -> Result<()> {
  if got != expected {
    return Err(Error::PreprocessedLength {
      feature,
      got,
      expected,
    });
  }
  Ok(())
}

/// Scans a caller-supplied preprocessed tensor for the first non-finite
/// (NaN/±∞) component (cf. `check_finite_output`, which classifies the same
/// defect on the MODEL-output side).
fn check_tensor_finite(feature: &'static str, values: &[f32]) -> Result<()> {
  if let Some(index) = values.iter().position(|v| !v.is_finite()) {
    return Err(Error::PreprocessedNonFinite { feature, index });
  }
  Ok(())
}

/// Validates the `[P]` attention mask is an exact NaFlex real/pad mask —
/// every entry exactly `0.0` or `1.0` (IEEE equality, so `-0.0` counts as
/// `0.0`; a NaN entry fails the domain check, which subsumes finiteness), the
/// `1.0`s a contiguous prefix, at least one real patch — and returns the
/// real-patch count (the prefix length). Exact comparison is deliberate: the
/// pipeline writes these constants literally, and an approximate check would
/// defeat the domain validation.
fn validate_mask(mask: &[f32]) -> Result<usize> {
  let mut num_real = 0usize;
  let mut in_pad = false;
  for (index, &value) in mask.iter().enumerate() {
    if value == 1.0 {
      if in_pad {
        return Err(Error::PreprocessedMaskOrder { index });
      }
      num_real += 1;
    } else if value == 0.0 {
      in_pad = true;
    } else {
      return Err(Error::PreprocessedMaskValue { index, value });
    }
  }
  if num_real == 0 {
    return Err(Error::PreprocessedMaskEmpty);
  }
  Ok(num_real)
}

/// Validates rows `num_real..` of a `[P · row_dim]` tensor are all-zero: the
/// NaFlex pipeline zero-fills pad rows and the module's parity evidence covers
/// only zero pads, so nonzero pad content is rejected fail-closed rather than
/// trusted to be masked out by the graph. Callers guarantee
/// `num_real · row_dim ≤ values.len()` (both bounded by the budget guard and
/// the mask length check).
fn validate_pad_rows(
  feature: &'static str,
  values: &[f32],
  num_real: usize,
  row_dim: usize,
) -> Result<()> {
  let pad_start = num_real * row_dim;
  if let Some(offset) = values[pad_start..].iter().position(|&v| v != 0.0) {
    return Err(Error::PreprocessedPadNonZero {
      feature,
      index: pad_start + offset,
    });
  }
  Ok(())
}

/// The structural (non-finiteness) subset of [`PreprocessedImage::try_new`]'s
/// validation — budget, lengths, mask shape, zero pads: exactly the invariants
/// the internal pipeline guarantees by construction, debug-asserted by
/// `PreprocessedImage::from_pipeline`.
fn validate_structural(
  pixel_values: &[f32],
  position_embeddings: &[f32],
  attention_mask: &[f32],
  max_num_patches: usize,
) -> Result<()> {
  validate_budget_and_lengths(
    pixel_values,
    position_embeddings,
    attention_mask,
    max_num_patches,
  )?;
  let num_real = validate_mask(attention_mask)?;
  validate_pad_rows(names::PIXEL_VALUES, pixel_values, num_real, PATCH_DIM)?;
  validate_pad_rows(
    names::POSITION_EMBEDDINGS,
    position_embeddings,
    num_real,
    EMBEDDING_DIM,
  )?;
  Ok(())
}

/// Validates a [`PreprocessedImage`]'s budget binding against the loaded
/// model's resolved `P`. Extracted so the classification is hermetically
/// testable without a loaded model (the `check_finite_output` pattern).
///
/// # Errors
/// [`Error::PatchBudgetMismatch`] carrying both budgets.
fn check_patch_budget(input: usize, model: usize) -> Result<()> {
  if input != model {
    return Err(Error::PatchBudgetMismatch { input, model });
  }
  Ok(())
}

/// Human-readable `shape dtype` rendering for [`Error::ContractMismatch`].
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("none", |d| d.as_str());
  format!("{shape:?} {dtype}")
}

#[cfg(test)]
mod tests;
