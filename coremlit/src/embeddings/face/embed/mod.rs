//! The CoreML face embedder: aligned template faces in, L2-normalised
//! embeddings out.
//!
//! # The embedder is pure, and preprocessing is DATA
//!
//! Channel order, scale, bias and tensor layout live in [`FaceModel`] — the
//! per-artifact manifest — and never in a constant at a call site. Issue #115's
//! census is the reason: six ArcFace-family artifacts use **four different
//! divisors, two channel orders and one per-channel mean**, and two "official"
//! releases of one model disagree about RGB vs BGR with no warning on either.
//! A wrong divisor costs `1 − cos ≈ 0.083` and a wrong channel order `0.151`,
//! against an ANE fp16 noise floor of `0.0015`. None of it raises an error;
//! all of it is silent degradation. Making it a value the caller supplies
//! alongside the weights is what keeps a second model from becoming a second
//! code path.
//!
//! | model | order | normalisation |
//! |---|---|---|
//! | `w600k_r50`, `glintr100` | RGB | `(x − 127.5) / 127.5` |
//! | ONNX-zoo ArcFace, OpenCV SFace | RGB | `(x − 128) / 128` (often fused) |
//! | AdaFace | **BGR** | `(x − 127.5) / 127.5` |
//! | FaceNet | RGB | `(x − 127.5) / 128` |
//! | dlib | RGB | `(x − [122.782, 117.001, 104.298]) / 256` |
//!
//! Every row of that table is expressible as [`Preprocessing`]'s `scale` plus a
//! per-channel `bias`.
//!
//! # Batch is the unit
//!
//! [`FaceEmbedder::embed`] takes a slice: a keyframe with N faces is ONE call,
//! whatever the graph's own batch dimension turns out to be. The capacity is
//! read off the loaded model's input contract at load
//! ([`FaceEmbedder::batch_capacity`]) and the slice is chunked to it, so a
//! batch-1 export and a batch-8 export are the same call site.

use std::path::Path;

use crate::{
  ComputeUnits, Model, MultiArray,
  embeddings::face::{
    align::{AlignedFace, TEMPLATE_SIZE},
    error::{BatchRow, ContractMismatch, Error, NonFiniteOutput, OutputShape, Result},
  },
};

/// The channel order a model's input tensor expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum ChannelOrder {
  /// Red, green, blue — the order [`AlignedFace`] stores.
  Rgb,
  /// Blue, green, red — OpenCV's order, and AdaFace's original checkpoints'.
  Bgr,
}

/// The axis order a model's input tensor expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum TensorLayout {
  /// `[batch, channel, height, width]` — the PyTorch/ONNX convention every
  /// ArcFace export in the census uses.
  Nchw,
  /// `[batch, height, width, channel]` — the TensorFlow convention.
  Nhwc,
}

/// One model's host-side preprocessing: `value = byte · scale + bias[channel]`.
///
/// `scale` and `bias` are in the MODEL's channel order, so a BGR model's
/// per-channel bias is written blue-first. Both forms in the module table
/// reduce to this: a divisor `d` and a mean `m` are `scale = 1/d`,
/// `bias = −m/d`.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Preprocessing {
  order: ChannelOrder,
  layout: TensorLayout,
  scale: f32,
  bias: [f32; 3],
}

impl Preprocessing {
  /// The ArcFace family's own preprocessing: RGB, NCHW, `(x − 127.5) / 127.5`
  /// — the `[−1, 1]` mapping `w600k_r50` and `glintr100` are trained and
  /// exported against.
  pub const ARCFACE: Self = Self {
    order: ChannelOrder::Rgb,
    layout: TensorLayout::Nchw,
    scale: 1.0 / 127.5,
    bias: [-1.0, -1.0, -1.0],
  };

  /// Preprocessing from its four parts.
  #[inline]
  pub const fn new(order: ChannelOrder, layout: TensorLayout, scale: f32, bias: [f32; 3]) -> Self {
    Self {
      order,
      layout,
      scale,
      bias,
    }
  }

  /// Preprocessing written as the census states it: a per-channel `mean`
  /// subtracted, then a `divisor`.
  ///
  /// `mean` and `divisor` are in the model's own channel order. Equivalent to
  /// [`Self::new`] with `scale = 1/divisor` and `bias = −mean/divisor`.
  #[inline]
  pub const fn from_mean_and_divisor(
    order: ChannelOrder,
    layout: TensorLayout,
    mean: [f32; 3],
    divisor: f32,
  ) -> Self {
    Self {
      order,
      layout,
      scale: 1.0 / divisor,
      bias: [-mean[0] / divisor, -mean[1] / divisor, -mean[2] / divisor],
    }
  }

  /// The channel order the model's input tensor expects.
  #[inline]
  pub const fn order(&self) -> ChannelOrder {
    self.order
  }

  /// The axis order the model's input tensor expects.
  #[inline]
  pub const fn layout(&self) -> TensorLayout {
    self.layout
  }

  /// The multiplier applied to each raw 0–255 byte.
  #[inline]
  pub const fn scale(&self) -> f32 {
    self.scale
  }

  /// The per-channel offset added after [`Self::scale`], in the model's own
  /// channel order.
  #[inline]
  pub const fn bias(&self) -> [f32; 3] {
    self.bias
  }
}

/// One face-embedding artifact's contract: what its input and output features
/// are called, how it wants its pixels, and how wide its embedding is.
///
/// A manifest is a VALUE, so wiring a second artifact with different
/// preprocessing is a different manifest at the same call site rather than a
/// second code path.
/// **No serde.** The feature names are `&'static str`, a compile-time contract
/// with the artifact rather than a runtime setting, and `Deserialize` cannot
/// produce a `&'static str` at all. The part that genuinely varies between
/// artifacts — [`Preprocessing`] — is serialisable on its own.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FaceModel {
  input: &'static str,
  output: &'static str,
  dim: usize,
  preprocessing: Preprocessing,
}

impl FaceModel {
  /// A manifest for an artifact with the given feature names and embedding
  /// width, preprocessed as [`Preprocessing::ARCFACE`].
  ///
  /// `dim` is 512 for every ArcFace-family artifact in issue #115's census.
  #[inline]
  pub const fn new(input: &'static str, output: &'static str, dim: usize) -> Self {
    Self {
      input,
      output,
      dim,
      preprocessing: Preprocessing::ARCFACE,
    }
  }

  /// The model's input feature name.
  #[inline]
  pub const fn input(&self) -> &'static str {
    self.input
  }

  /// The model's output feature name.
  #[inline]
  pub const fn output(&self) -> &'static str {
    self.output
  }

  /// The embedding width the artifact produces.
  #[inline]
  pub const fn dim(&self) -> usize {
    self.dim
  }

  /// The artifact's host-side preprocessing.
  #[inline]
  pub const fn preprocessing(&self) -> Preprocessing {
    self.preprocessing
  }

  /// Builder form of [`Self::set_preprocessing`].
  #[must_use]
  #[inline]
  pub const fn with_preprocessing(mut self, preprocessing: Preprocessing) -> Self {
    self.set_preprocessing(preprocessing);
    self
  }

  /// Sets [`Self::preprocessing`] in place.
  #[inline]
  pub const fn set_preprocessing(&mut self, preprocessing: Preprocessing) -> &mut Self {
    self.preprocessing = preprocessing;
    self
  }
}

/// Default [`FaceEmbedderOptions::compute`]: [`ComputeUnits::All`].
///
/// **Not a measured pin, unlike `siglip`'s and `clap`'s.** Those defaults were
/// chosen after characterising every arm on a staged artifact; this crate
/// stages no face artifact yet (issue #115, and this module's own doc), so
/// there is nothing to characterise and the honest default is CoreML's own
/// planner choice. Issue #115's parity census predicts what the measurement
/// will find — an IResNet's 24 residual `Add` chains put the ANE's fp16 arm at
/// `1 − cos ≈ 0.0015` typical against a GPU that accumulates in fp32 — so the
/// arm is expected to be usable and the default is expected to survive. Expected
/// is not measured: characterise before relying on it.
pub const DEFAULT_FACE_COMPUTE: ComputeUnits = ComputeUnits::All;

#[cfg(feature = "serde")]
fn default_face_compute() -> ComputeUnits {
  DEFAULT_FACE_COMPUTE
}

/// Construction options for [`FaceEmbedder`] (rust-options-pattern): a single
/// `compute` knob with one source of truth shared by `const new`/`Default`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FaceEmbedderOptions {
  #[cfg_attr(
    feature = "serde",
    serde(
      default = "default_face_compute",
      with = "crate::embeddings::face::compute_units_serde"
    )
  )]
  compute: ComputeUnits,
}

impl Default for FaceEmbedderOptions {
  fn default() -> Self {
    Self::new()
  }
}

impl FaceEmbedderOptions {
  /// Options matching the module default: [`DEFAULT_FACE_COMPUTE`].
  #[inline]
  pub const fn new() -> Self {
    Self {
      compute: DEFAULT_FACE_COMPUTE,
    }
  }

  /// Which hardware CoreML may schedule the graph on.
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

/// One face's L2-normalised embedding.
///
/// Unit norm, so [`Self::cosine`] is a dot product and a threshold means the
/// same thing for every face. The width is the ARTIFACT's
/// ([`FaceModel::dim`]), not a code constant — 512 for every ArcFace-family
/// model in issue #115's census, but a second family is a different manifest,
/// not a different type.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceEmbedding {
  values: Box<[f32]>,
}

impl FaceEmbedding {
  /// The embedding width.
  #[inline]
  pub fn dim(&self) -> usize {
    self.values.len()
  }

  /// The unit-norm components.
  #[inline]
  pub fn as_slice(&self) -> &[f32] {
    &self.values
  }

  /// An owned copy of the components.
  #[inline]
  pub fn to_vec(&self) -> Vec<f32> {
    self.values.to_vec()
  }

  /// The dot product with `other`, which for two unit vectors of equal width
  /// is their cosine.
  ///
  /// Returns `0.0` when the widths differ — two embeddings from different
  /// artifacts are not comparable, and a panic in a similarity function would
  /// be a poor way to say so.
  #[inline]
  #[must_use]
  pub fn dot(&self, other: &Self) -> f32 {
    if self.values.len() != other.values.len() {
      return 0.0;
    }
    self
      .values
      .iter()
      .zip(other.values.iter())
      .map(|(x, y)| x * y)
      .sum()
  }

  /// The cosine similarity with `other` — an alias for [`Self::dot`], since
  /// both operands are unit norm by construction.
  #[inline]
  #[must_use]
  pub fn cosine(&self, other: &Self) -> f32 {
    self.dot(other)
  }
}

/// The CoreML face embedder: a batch of [`AlignedFace`]s in, one
/// [`FaceEmbedding`] each out.
///
/// `&self` inference — the per-call input tensor is local, so fan-out means one
/// embedder per worker over a `Send` (but deliberately `!Sync`) [`Model`],
/// matching every other kit in this crate.
#[derive(Debug)]
pub struct FaceEmbedder {
  model: Model,
  manifest: FaceModel,
  /// The graph's own batch dimension, read from the input contract at load.
  batch: usize,
}

impl FaceEmbedder {
  /// Loads a compiled `.mlmodelc` and binds it to `manifest`.
  ///
  /// The manifest's feature names and embedding width are reconciled against
  /// the model's declared contract here, so a manifest that names the wrong
  /// features, or claims the wrong width, fails at load rather than producing
  /// a plausible-looking wrong vector.
  ///
  /// # Errors
  /// [`Error::Load`] if CoreML rejects the model;
  /// [`Error::ContractMismatch`] if the model declares no feature by the
  /// manifest's name, or if its input/output shapes are not a batch of
  /// `3 × 112 × 112` and a batch of [`FaceModel::dim`].
  pub fn load(
    model_path: impl AsRef<Path>,
    manifest: FaceModel,
    options: FaceEmbedderOptions,
  ) -> Result<Self> {
    let model = Model::load(model_path, options.compute())?;
    let batch = {
      let description = model.description();
      let input = description.input(manifest.input()).ok_or_else(|| {
        Error::ContractMismatch(ContractMismatch::new(
          manifest.input().to_string(),
          "a declared input feature".to_string(),
          format!("inputs {:?}", feature_names(description.inputs())),
        ))
      })?;
      let batch =
        resolve_batch(input.shape(), manifest.preprocessing().layout()).ok_or_else(|| {
          Error::ContractMismatch(ContractMismatch::new(
            manifest.input().to_string(),
            format!(
              "{:?} shaped [n, 3, {TEMPLATE_SIZE}, {TEMPLATE_SIZE}] (or without the batch axis)",
              manifest.preprocessing().layout()
            ),
            format!("{:?}", input.shape()),
          ))
        })?;
      let output = description.output(manifest.output()).ok_or_else(|| {
        Error::ContractMismatch(ContractMismatch::new(
          manifest.output().to_string(),
          "a declared output feature".to_string(),
          format!("outputs {:?}", feature_names(description.outputs())),
        ))
      })?;
      check_output_shape(output.shape(), batch, manifest.dim()).map_err(|actual| {
        Error::ContractMismatch(ContractMismatch::new(
          manifest.output().to_string(),
          format!("[{batch}, {}] (or [{}])", manifest.dim(), manifest.dim()),
          actual,
        ))
      })?;
      batch
    };
    Ok(Self {
      model,
      manifest,
      batch,
    })
  }

  /// Loads with [`FaceEmbedderOptions::new`].
  ///
  /// # Errors
  /// As [`Self::load`].
  pub fn from_file(model_path: impl AsRef<Path>, manifest: FaceModel) -> Result<Self> {
    Self::load(model_path, manifest, FaceEmbedderOptions::new())
  }

  /// The manifest this embedder was bound to.
  #[inline]
  pub const fn manifest(&self) -> &FaceModel {
    &self.manifest
  }

  /// The graph's own batch dimension, resolved from its input contract at load.
  ///
  /// [`Self::embed`] chunks any slice to this, so it is a throughput fact
  /// rather than a call-site constraint.
  #[inline]
  pub const fn batch_capacity(&self) -> usize {
    self.batch
  }

  /// The embedding width — [`FaceModel::dim`], reconciled against the model at
  /// load.
  #[inline]
  pub const fn dim(&self) -> usize {
    self.manifest.dim()
  }

  /// Embeds a batch of aligned faces, one [`FaceEmbedding`] per input, in
  /// order.
  ///
  /// An empty slice yields an empty vector without touching the model. Longer
  /// slices are chunked to [`Self::batch_capacity`]; a short final chunk is
  /// zero-padded and the padding rows are discarded, so the result length
  /// always equals `faces.len()`.
  ///
  /// # Errors
  /// [`Error::Tensor`] / [`Error::Prediction`] on a tensor or CoreML failure;
  /// [`Error::OutputShape`] if a predicted tensor's shape diverges from the
  /// contract resolved at load; [`Error::NonFiniteOutput`] if the model emits
  /// a NaN or infinite component; [`Error::EmbeddingZero`] if a (finite)
  /// output row has zero magnitude and cannot be normalised.
  pub fn embed(&self, faces: &[AlignedFace]) -> Result<Vec<FaceEmbedding>> {
    let mut out = Vec::with_capacity(faces.len());
    for (chunk_index, chunk) in faces.chunks(self.batch).enumerate() {
      let rows = self.predict_chunk(chunk, chunk_index * self.batch)?;
      out.extend(rows);
    }
    Ok(out)
  }

  /// Predicts one chunk of at most [`Self::batch_capacity`] faces.
  ///
  /// `first_row` is the chunk's offset into the caller's slice, so every error
  /// names the caller's own index rather than a position inside a chunk the
  /// caller never saw.
  fn predict_chunk(&self, chunk: &[AlignedFace], first_row: usize) -> Result<Vec<FaceEmbedding>> {
    let dim = self.manifest.dim();
    let tensor = self.build_input(chunk)?;
    let mut outputs = self
      .model
      .predict_with(&[(self.manifest.input(), &tensor)])?;
    let features = outputs
      .take(self.manifest.output())
      .ok_or_else(|| crate::PredictionError::MissingOutput(self.manifest.output().to_string()))?;
    let expected = vec![self.batch, dim];
    if features.count() != self.batch * dim {
      return Err(Error::OutputShape(OutputShape::new(
        features.shape().to_vec(),
        expected,
      )));
    }

    let mut flat = vec![0.0f32; self.batch * dim];
    features.copy_into::<f32>(&mut flat)?;
    let mut rows = Vec::with_capacity(chunk.len());
    for (offset, row) in flat.chunks_exact(dim).take(chunk.len()).enumerate() {
      rows.push(normalise_row(row, first_row + offset)?);
    }
    Ok(rows)
  }

  /// Builds the `[batch, …]` input tensor for one chunk, zero-padding the tail
  /// rows a short chunk leaves.
  fn build_input(&self, chunk: &[AlignedFace]) -> Result<MultiArray> {
    let preprocessing = self.manifest.preprocessing();
    let pixels = TEMPLATE_SIZE * TEMPLATE_SIZE;
    let mut data = vec![0.0f32; self.batch * 3 * pixels];
    for (row, face) in chunk.iter().enumerate() {
      write_row(
        &mut data[row * 3 * pixels..(row + 1) * 3 * pixels],
        face,
        preprocessing,
      );
    }
    let shape = match preprocessing.layout() {
      TensorLayout::Nchw => [self.batch, 3, TEMPLATE_SIZE, TEMPLATE_SIZE],
      TensorLayout::Nhwc => [self.batch, TEMPLATE_SIZE, TEMPLATE_SIZE, 3],
    };
    Ok(MultiArray::from_slice(&shape, &data)?)
  }
}

/// The declared feature names, for a contract-mismatch message.
fn feature_names(features: &[crate::FeatureInfo]) -> Vec<&str> {
  features.iter().map(crate::FeatureInfo::name).collect()
}

/// The batch dimension an input `shape` declares, or `None` when the shape is
/// not a template face.
///
/// Accepts three shapes, all of which real ArcFace exports use: the batched
/// rank-4 form, the unbatched rank-3 form (batch 1), and an EMPTY shape — the
/// legacy `neuralNetwork` specification leaves input shapes undeclared, and
/// refusing those would refuse a whole artifact format on the strength of
/// metadata the format does not carry. An empty shape resolves to batch 1 and
/// is caught instead at predict time by the output-shape check.
fn resolve_batch(shape: &[usize], layout: TensorLayout) -> Option<usize> {
  let (channels, height, width) = match layout {
    TensorLayout::Nchw => (0usize, 1usize, 2usize),
    TensorLayout::Nhwc => (2usize, 0usize, 1usize),
  };
  let matches = |dims: &[usize]| {
    dims[channels] == 3 && dims[height] == TEMPLATE_SIZE && dims[width] == TEMPLATE_SIZE
  };
  match shape.len() {
    0 => Some(1),
    3 if matches(shape) => Some(1),
    // A declared batch of zero is not a batch: `embed` would divide the work
    // into chunks of zero and never terminate, so it is a contract mismatch
    // rather than a capacity.
    4 if shape[0] > 0 && matches(&shape[1..]) => Some(shape[0]),
    _ => None,
  }
}

/// Checks an output `shape` against the resolved contract, returning the
/// rendered actual shape on mismatch.
///
/// Like [`resolve_batch`], an empty shape is accepted: a legacy `neuralNetwork`
/// artifact declares none, and the predicted tensor is checked on every call
/// regardless.
fn check_output_shape(
  shape: &[usize],
  batch: usize,
  dim: usize,
) -> core::result::Result<(), String> {
  let ok = match shape.len() {
    0 => true,
    1 => shape[0] == dim,
    2 => shape[0] == batch && shape[1] == dim,
    _ => false,
  };
  if ok {
    Ok(())
  } else {
    Err(format!("{shape:?}"))
  }
}

/// Writes one aligned face into `row` as `3 · 112 · 112` preprocessed floats.
fn write_row(row: &mut [f32], face: &AlignedFace, preprocessing: Preprocessing) {
  let pixels = TEMPLATE_SIZE * TEMPLATE_SIZE;
  let source = face.pixels();
  let (scale, bias) = (preprocessing.scale(), preprocessing.bias());
  for pixel in 0..pixels {
    for (channel, offset) in bias.iter().enumerate() {
      // `channel` indexes the MODEL's channel; `source_channel` is where that
      // channel's byte lives in the RGB-interleaved template.
      let source_channel = match preprocessing.order() {
        ChannelOrder::Rgb => channel,
        ChannelOrder::Bgr => 2 - channel,
      };
      let value = f32::from(source[pixel * 3 + source_channel]).mul_add(scale, *offset);
      let index = match preprocessing.layout() {
        TensorLayout::Nchw => channel * pixels + pixel,
        TensorLayout::Nhwc => pixel * 3 + channel,
      };
      row[index] = value;
    }
  }
}

/// L2-normalises one model output row, classifying a non-finite component and
/// a zero magnitude separately.
fn normalise_row(row: &[f32], index: usize) -> Result<FaceEmbedding> {
  if let Some(component) = row.iter().position(|v| !v.is_finite()) {
    return Err(Error::NonFiniteOutput(NonFiniteOutput::new(
      index, component,
    )));
  }
  let norm = row.iter().map(|v| v * v).sum::<f32>().sqrt();
  if norm == 0.0 || !norm.is_finite() {
    return Err(Error::EmbeddingZero(BatchRow::new(index)));
  }
  Ok(FaceEmbedding {
    values: row.iter().map(|v| v / norm).collect(),
  })
}

#[cfg(test)]
mod tests;
