//! The siglip [`ImageEmbedder`]: the NaFlex host-side preprocessing (the private
//! `preprocess` submodule) around the fp16 CoreML vision graph, with L2
//! normalization applied in Rust.

mod preprocess;

use std::path::Path;

use crate::{ComputeUnits, DataType, Model, ModelDescription, MultiArray};

use crate::embeddings::siglip::{
  embedding::{EMBEDDING_DIM, Embedding, check_finite_output},
  error::{Error, Result},
  image::preprocess::{PATCH_DIM, parse_base_pos_grid, preprocess_image},
};

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
    let inputs = preprocess_image(
      image.data(),
      image.width(),
      image.height(),
      &self.base_pos_embed,
      self.max_num_patches,
    )?;

    let pixel_values =
      MultiArray::from_slice(&[1, self.max_num_patches, PATCH_DIM], &inputs.pixel_values)?;
    let position_embeddings = MultiArray::from_slice(
      &[1, self.max_num_patches, EMBEDDING_DIM],
      &inputs.position_embeddings,
    )?;
    let attention_mask =
      MultiArray::from_slice(&[1, self.max_num_patches], &inputs.attention_mask)?;

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

/// Human-readable `shape dtype` rendering for [`Error::ContractMismatch`].
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("none", |d| d.as_str());
  format!("{shape:?} {dtype}")
}

#[cfg(test)]
mod tests;
