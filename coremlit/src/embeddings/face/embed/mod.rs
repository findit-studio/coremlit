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
  ComputeUnits, DataType, Model, MultiArray,
  embeddings::face::{
    align::{AlignedFace, TEMPLATE_SIZE},
    error::{
      BatchRow, ContractMismatch, EmbeddingSpaceField, Error, IncomparableEmbeddings,
      NonFiniteOutput, OutputElementCount, OutputShape, Result,
    },
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

/// One face's L2-normalised embedding, carrying the space it belongs to.
///
/// Unit norm, so [`Self::cosine`] is a dot product and a threshold means the
/// same thing for every face. The width is the ARTIFACT's
/// ([`FaceModel::dim`]), not a code constant — 512 for every ArcFace-family
/// model in issue #115's census, but a second family is a different manifest,
/// not a different type.
///
/// # The space travels with the vector
///
/// A cosine is only meaningful between two vectors in the same space, and the
/// widths agreeing does not establish that: two 512-wide ArcFace-family
/// artifacts, or one artifact fed BGR where it was trained on RGB, produce
/// vectors whose dot product lands in `[−1, 1]` looking exactly like a
/// measurement. So each embedding carries the [`FaceModel`] that produced it
/// and [`Self::dot`] REFUSES a pair that disagrees.
///
/// **A value here is produced, never assembled.** There is no public
/// constructor: the only way to obtain a `FaceEmbedding` is
/// [`FaceEmbedder::embed`], which stamps its own bound manifest onto every row
/// it returns. So the space is not something a caller states about a vector —
/// it is what the embedder that computed the vector was loaded against.
///
/// # What this takes from `audio::speaker::calibrate`, and what it does not
///
/// The `SpeakerToken` work converged, over six rounds, on making a wrong
/// pairing **unrepresentable** rather than refused: the caller's key type was
/// removed from the cohort surface entirely, because the defect was never the
/// road that reached it but that an identity could be *resolved* from
/// caller-owned state at all.
///
/// Taken: the identity rides on the value rather than being an argument at the
/// call site, no caller-owned value takes part in deciding it, and the part
/// this crate cannot refute is stated below instead of claimed away.
///
/// **Not taken: a minted, process-unique token, and the reason is a
/// difference in the two problems rather than a lighter standard.** A cohort is
/// one object, so minting inside it costs nothing. A face embedding space is
/// legitimately produced by MORE than one producer — `&self` inference means
/// fan-out is one [`FaceEmbedder`] per worker over the same artifact (see that
/// type's doc), and a per-load token would refuse the cross-worker comparisons
/// those workers exist to make. Turning a silent wrong answer into a loud
/// wrong answer is not the precedent's cure. There is also no lookup here to
/// remove: `calibrate`'s defect was a question that could be answered twice
/// differently, and [`Self::dot`] asks nothing of caller-owned state — it
/// compares two values' own recorded manifests, by bit pattern, so a NaN
/// scale still names one consistent space rather than a space unequal to
/// itself.
///
/// **The residual, stated.** Equality of manifests is the strongest evidence a
/// sans-I/O crate holds: `coremlit` does not hold the weights, so two DISTINCT
/// artifacts declaring the same feature names, width and preprocessing are one
/// space as far as this type can see, and their cosine is returned rather than
/// refused. That is the same shape as `calibrate`'s own stated residual —
/// `Enrolled::new` *claims* a probe belongs to a speaker and no type in this
/// crate can refute it. [`AlignedFace::from_template_pixels`] carries the
/// matching hole on the pixel side, and says so.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceEmbedding {
  /// The unit-norm components. Always `space.dim()` of them: the only
  /// constructor fills this from a row the manifest's own width cut.
  values: Box<[f32]>,
  /// The manifest of the embedder that produced this vector.
  space: FaceModel,
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

  /// The manifest of the embedder that produced this embedding — the space it
  /// lives in.
  ///
  /// Readable so a caller storing embeddings can tell which artifact a stored
  /// vector came from. Reading it cannot forge one: [`FaceEmbedding`] has no
  /// public constructor, so a [`FaceModel`] a caller builds can never end up
  /// stamped on a vector this crate did not compute.
  #[inline]
  pub const fn manifest(&self) -> &FaceModel {
    &self.space
  }

  /// The dot product with `other`, which for two unit vectors in one space is
  /// their cosine.
  ///
  /// # Errors
  /// [`Error::IncomparableEmbeddings`] if the two came from different model or
  /// preprocessing spaces, naming the first manifest field that differs.
  ///
  /// **Fallible rather than a sentinel.** This used to return `0.0` for a
  /// width mismatch, which is also what a measured orthogonal pair returns —
  /// so a caller could not tell an incompatible model migration from a face
  /// that did not match. A width mismatch is now one arm of the space check,
  /// reported as [`EmbeddingSpaceField::Dim`], and the arm nothing could
  /// report before — equal widths, different spaces — is the rest of it.
  #[inline]
  pub fn dot(&self, other: &Self) -> Result<f32> {
    if let Some(field) = space_difference(&self.space, &other.space) {
      return Err(Error::IncomparableEmbeddings(IncomparableEmbeddings::new(
        field,
      )));
    }
    Ok(
      self
        .values
        .iter()
        .zip(other.values.iter())
        .map(|(x, y)| x * y)
        .sum(),
    )
  }

  /// The cosine similarity with `other` — an alias for [`Self::dot`], since
  /// both operands are unit norm by construction.
  ///
  /// # Errors
  /// As [`Self::dot`].
  #[inline]
  pub fn cosine(&self, other: &Self) -> Result<f32> {
    self.dot(other)
  }
}

/// The first field of two manifests that puts their embeddings in different
/// spaces, or `None` when they name one space.
///
/// The `f32`s are compared by BIT PATTERN rather than by `==`. A manifest with
/// a NaN scale is a broken manifest, but it is ONE broken manifest: under `==`
/// it would fail to equal itself and every comparison involving it — including
/// an embedding against its own twin — would be refused for a reason that has
/// nothing to do with the two embeddings. `to_bits` makes a manifest name the
/// same space as itself whatever it holds, which is the only property this
/// check needs from it.
///
/// Every field is compared, not short-circuited — the array is built before
/// `find_map` walks it — so the order decides only WHICH field is named when
/// several differ at once. `Dim` leads because it is the one a caller is most
/// likely to have caused and the only one the old width check could see.
fn space_difference(left: &FaceModel, right: &FaceModel) -> Option<EmbeddingSpaceField> {
  let (lp, rp) = (left.preprocessing(), right.preprocessing());
  let bits = |b: [f32; 3]| b.map(f32::to_bits);
  [
    (EmbeddingSpaceField::Dim, left.dim() != right.dim()),
    (
      EmbeddingSpaceField::InputFeature,
      left.input() != right.input(),
    ),
    (
      EmbeddingSpaceField::OutputFeature,
      left.output() != right.output(),
    ),
    (EmbeddingSpaceField::ChannelOrder, lp.order() != rp.order()),
    (
      EmbeddingSpaceField::TensorLayout,
      lp.layout() != rp.layout(),
    ),
    (
      EmbeddingSpaceField::PreprocessingScale,
      lp.scale().to_bits() != rp.scale().to_bits(),
    ),
    (
      EmbeddingSpaceField::PreprocessingBias,
      bits(lp.bias()) != bits(rp.bias()),
    ),
  ]
  .into_iter()
  .find_map(|(field, differs)| differs.then_some(field))
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
  /// The graph's own batch dimension AND declared rank, read from the input
  /// contract at load. The rank is carried, not just the capacity: a model
  /// that declares the unbatched rank-3 form has to be fed a rank-3 tensor.
  input: InputContract,
  /// The output form the graph declared, so a predicted tensor is checked
  /// against the axes it promised and not merely against an element count.
  output: OutputContract,
}

impl FaceEmbedder {
  /// Loads a compiled `.mlmodelc` and binds it to `manifest`.
  ///
  /// The manifest's feature names and embedding width are reconciled against
  /// the model's declared contract here, so a manifest that names the wrong
  /// features, or claims the wrong width, fails at load rather than producing
  /// a plausible-looking wrong vector.
  ///
  /// Both features must be `float32` MULTI-ARRAYS. Inference supplies and
  /// extracts nothing else, so an f16 export — or an `ImageType` feature,
  /// which carries no shape and no element type at all and is what both
  /// third-party CoreML ArcFace builds this module's doc surveys declare — is
  /// refused here rather than loading clean and failing every prediction.
  ///
  /// # Errors
  /// [`Error::Load`] if CoreML rejects the model;
  /// [`Error::ContractMismatch`] if the model declares no feature by the
  /// manifest's name, if either feature is not a `float32` multi-array, or if
  /// its input/output shapes are not a batch of `3 × 112 × 112` and a batch of
  /// [`FaceModel::dim`].
  pub fn load(
    model_path: impl AsRef<Path>,
    manifest: FaceModel,
    options: FaceEmbedderOptions,
  ) -> Result<Self> {
    let model = Model::load(model_path, options.compute())?;
    let (input_contract, output_contract) = {
      let description = model.description();
      let input = description.input(manifest.input()).ok_or_else(|| {
        Error::ContractMismatch(ContractMismatch::new(
          manifest.input().to_string(),
          "a declared input feature".to_string(),
          format!("inputs {:?}", feature_names(description.inputs())),
        ))
      })?;
      let contract = resolve_input_contract(
        input.shape(),
        input.data_type(),
        manifest.preprocessing().layout(),
      )
      .ok_or_else(|| {
        Error::ContractMismatch(ContractMismatch::new(
          manifest.input().to_string(),
          format!(
            "{:?} shaped [n, 3, {TEMPLATE_SIZE}, {TEMPLATE_SIZE}] (or without the batch axis) \
             float32",
            manifest.preprocessing().layout()
          ),
          describe(input.shape(), input.data_type()),
        ))
      })?;
      let batch = contract.batch();
      let output = description.output(manifest.output()).ok_or_else(|| {
        Error::ContractMismatch(ContractMismatch::new(
          manifest.output().to_string(),
          "a declared output feature".to_string(),
          format!("outputs {:?}", feature_names(description.outputs())),
        ))
      })?;
      let declared =
        check_output_contract(output.shape(), output.data_type(), batch, manifest.dim()).map_err(
          |actual| {
            Error::ContractMismatch(ContractMismatch::new(
              manifest.output().to_string(),
              format!(
                "[{batch}, {}] (or [{}] for a batch-one graph) float32",
                manifest.dim(),
                manifest.dim()
              ),
              actual,
            ))
          },
        )?;
      (contract, declared)
    };
    Ok(Self {
      model,
      manifest,
      input: input_contract,
      output: output_contract,
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
    self.input.batch
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
  /// [`Error::OutputShape`] if a predicted tensor's axes diverge from the
  /// contract resolved at load, or [`Error::OutputElementCount`] if only its
  /// element count does; [`Error::NonFiniteOutput`] if the model emits
  /// a NaN or infinite component; [`Error::EmbeddingZero`] if a (finite)
  /// output row has zero magnitude and cannot be normalised.
  pub fn embed(&self, faces: &[AlignedFace]) -> Result<Vec<FaceEmbedding>> {
    let mut out = Vec::with_capacity(faces.len());
    let batch = self.input.batch;
    for (chunk_index, chunk) in faces.chunks(batch).enumerate() {
      let rows = self.predict_chunk(chunk, chunk_index * batch)?;
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
    let batch = self.input.batch;
    check_predicted_shape(features.shape(), features.count(), self.output, batch, dim)?;

    let mut flat = vec![0.0f32; batch * dim];
    features.copy_into::<f32>(&mut flat)?;
    let mut rows = Vec::with_capacity(chunk.len());
    for (offset, row) in flat.chunks_exact(dim).take(chunk.len()).enumerate() {
      rows.push(normalise_row(row, first_row + offset, &self.manifest)?);
    }
    Ok(rows)
  }

  /// Builds the `[batch, …]` input tensor for one chunk, zero-padding the tail
  /// rows a short chunk leaves.
  fn build_input(&self, chunk: &[AlignedFace]) -> Result<MultiArray> {
    let preprocessing = self.manifest.preprocessing();
    let pixels = TEMPLATE_SIZE * TEMPLATE_SIZE;
    let mut data = vec![0.0f32; self.input.batch * 3 * pixels];
    for (row, face) in chunk.iter().enumerate() {
      write_row(
        &mut data[row * 3 * pixels..(row + 1) * 3 * pixels],
        face,
        preprocessing,
      );
    }
    let shape = input_shape(self.input, preprocessing.layout());
    Ok(MultiArray::from_slice(&shape, &data)?)
  }
}

/// The shape of the tensor [`FaceEmbedder::build_input`] hands the graph, for
/// a contract resolved at load.
///
/// The rank is the CONTRACT's, not a constant. A graph that declares
/// `[3, 112, 112]` is handed `[3, 112, 112]`; only a graph that declared the
/// batch axis is handed one.
fn input_shape(contract: InputContract, layout: TensorLayout) -> Vec<usize> {
  let face = match layout {
    TensorLayout::Nchw => [3, TEMPLATE_SIZE, TEMPLATE_SIZE],
    TensorLayout::Nhwc => [TEMPLATE_SIZE, TEMPLATE_SIZE, 3],
  };
  match contract.rank {
    InputRank::Unbatched => face.to_vec(),
    InputRank::Batched => {
      let mut shape = Vec::with_capacity(1 + face.len());
      shape.push(contract.batch);
      shape.extend_from_slice(&face);
      shape
    }
  }
}

/// The declared feature names, for a contract-mismatch message.
fn feature_names(features: &[crate::FeatureInfo]) -> Vec<&str> {
  features.iter().map(crate::FeatureInfo::name).collect()
}

/// Whether a model's input feature declares the batch axis.
///
/// Kept, rather than collapsed into the numeric capacity, because it decides
/// the RANK of the tensor [`input_shape`] then builds. Resolving
/// `[3, 112, 112]` to "batch 1" and building `[1, 3, 112, 112]` from it means
/// a model that loads as supported fails every prediction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum InputRank {
  /// `[n, 3, 112, 112]` / `[n, 112, 112, 3]` — the batch axis is declared.
  Batched,
  /// `[3, 112, 112]` / `[112, 112, 3]` — no batch axis, so capacity 1.
  Unbatched,
}

/// The input contract resolved from a model's declared feature at load.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct InputContract {
  /// How many faces one prediction consumes.
  batch: usize,
  /// The rank the graph declared, and therefore the rank it must be fed.
  rank: InputRank,
}

impl InputContract {
  /// How many faces one prediction consumes.
  #[inline]
  const fn batch(&self) -> usize {
    self.batch
  }
}

/// The input contract an input feature declares, or `None` when it is not a
/// `float32` multi-array holding a template face.
///
/// Accepts three shapes, all of which real ArcFace exports use: the batched
/// rank-4 form, the unbatched rank-3 form (batch 1), and an EMPTY shape — the
/// legacy `neuralNetwork` specification leaves input shapes undeclared, and
/// refusing those would refuse a whole artifact format on the strength of
/// metadata the format does not carry. An undeclared shape resolves to a
/// batch-one BATCHED contract — the rank an export that declares nothing
/// overwhelmingly means — and is caught instead at predict time by the
/// output-shape check.
///
/// The dtype is not decoration. `None` means the feature is not a multi-array
/// AT ALL — an `ImageType` input reports no shape and no element type, which
/// is what both third-party CoreML ArcFace builds the module doc surveys
/// declare — and anything but `float32` is a tensor
/// [`FaceEmbedder::build_input`] cannot supply. Both used to load clean and
/// then fail every prediction.
fn resolve_input_contract(
  shape: &[usize],
  dtype: Option<DataType>,
  layout: TensorLayout,
) -> Option<InputContract> {
  if dtype != Some(DataType::F32) {
    return None;
  }
  let (channels, height, width) = match layout {
    TensorLayout::Nchw => (0usize, 1usize, 2usize),
    TensorLayout::Nhwc => (2usize, 0usize, 1usize),
  };
  let matches = |dims: &[usize]| {
    dims[channels] == 3 && dims[height] == TEMPLATE_SIZE && dims[width] == TEMPLATE_SIZE
  };
  match shape.len() {
    0 => Some(InputContract {
      batch: 1,
      rank: InputRank::Batched,
    }),
    3 if matches(shape) => Some(InputContract {
      batch: 1,
      rank: InputRank::Unbatched,
    }),
    // A declared batch of zero is not a batch: `embed` would divide the work
    // into chunks of zero and never terminate, so it is a contract mismatch
    // rather than a capacity.
    4 if shape[0] > 0 && matches(&shape[1..]) => Some(InputContract {
      batch: shape[0],
      rank: InputRank::Batched,
    }),
    _ => None,
  }
}

/// The output form a model declared — and therefore the EXACT shape its
/// predicted tensor must have.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutputContract {
  /// The model declared `[batch, dim]`.
  Batched,
  /// The model declared `[dim]`, which only a batch-one graph can mean.
  Flat,
  /// The model declared no shape at all (a legacy `neuralNetwork`). Either
  /// form is then legitimate at predict time — and only those two.
  Undeclared,
}

/// Checks an output feature against the resolved contract, returning the form
/// it declared, or the rendered actual feature on mismatch.
///
/// Like [`resolve_input_contract`], an empty shape is accepted: a legacy
/// `neuralNetwork` artifact declares none, and the predicted tensor is checked
/// on every call regardless. A `[dim]` shape is accepted only for a batch-one
/// graph — against a batch of 4 it is not a shorthand, it is a contradiction.
fn check_output_contract(
  shape: &[usize],
  dtype: Option<DataType>,
  batch: usize,
  dim: usize,
) -> core::result::Result<OutputContract, String> {
  if dtype != Some(DataType::F32) {
    return Err(describe(shape, dtype));
  }
  let resolved = match shape.len() {
    0 => Some(OutputContract::Undeclared),
    1 if shape[0] == dim && batch == 1 => Some(OutputContract::Flat),
    2 if shape[0] == batch && shape[1] == dim => Some(OutputContract::Batched),
    _ => None,
  };
  resolved.ok_or_else(|| describe(shape, dtype))
}

/// Human-readable `shape dtype` rendering for [`ContractMismatch`].
fn describe(shape: &[usize], dtype: Option<DataType>) -> String {
  let dtype = dtype.map_or("not a multi-array", |d| d.as_str());
  format!("{shape:?} {dtype}")
}

/// Checks a PREDICTED tensor against the contract resolved at load — its AXES,
/// not merely its element count.
///
/// A count-only check is close to no check here. `[dim, batch]` holds exactly
/// as many elements as `[batch, dim]`, so it passes, and the
/// `chunks_exact(dim)` that follows then slices across the wrong axis: every
/// returned embedding is a mixture of components from different faces,
/// unit-norm and plausible, and nothing downstream — no shape check, no
/// finiteness scan, no cosine — can tell. So the shape must equal the resolved
/// contract exactly, with the bare `[dim]` form allowed only where the
/// contract really is batch-one.
///
/// The count is still checked alongside it: [`MultiArray::count`] is CoreML's
/// own answer rather than a product of the cached shape, and the copy that
/// follows sizes its destination from the contract. It gets an error of its
/// OWN, because it is a different failure: with the axes equal there is no
/// shape mismatch to report, and [`Error::OutputShape`] could only report one
/// by naming the same vector twice.
fn check_predicted_shape(
  shape: &[usize],
  count: usize,
  contract: OutputContract,
  batch: usize,
  dim: usize,
) -> Result<()> {
  let batched = [batch, dim];
  let flat = [dim];
  let expected: &[usize] = match contract {
    OutputContract::Batched => &batched,
    OutputContract::Flat => &flat,
    // The graph promised nothing, so either form is honest — but it still has
    // to be one of the two.
    OutputContract::Undeclared if batch == 1 && shape == flat => &flat,
    OutputContract::Undeclared => &batched,
  };
  if shape != expected {
    return Err(Error::OutputShape(OutputShape::new(
      shape.to_vec(),
      expected.to_vec(),
    )));
  }
  if count != batch * dim {
    return Err(Error::OutputElementCount(OutputElementCount::new(
      count,
      batch * dim,
    )));
  }
  Ok(())
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
///
/// The squared norm accumulates in `f64`, and the division happens there too.
/// In `f32` it could not: `v * v` overflows to infinity for a large component
/// and underflows to zero for a small one, and BOTH used to be reported as
/// [`Error::EmbeddingZero`] — "this row has no direction" — for a row with a
/// perfectly good direction. Nothing in the contract says an artifact's
/// pre-normalisation output is near unit scale, so its magnitude is the
/// model's business and not a reason to refuse it.
///
/// In `f64` the accumulator cannot leave the type: every component is a finite
/// `f32` by the scan above, so each square is at most `f32::MAX²` (≈1.2e77)
/// and even a very wide embedding sums far below `f64::MAX`, while the
/// smallest nonzero `f32` squares to ≈2e-90 — comfortably normal. The norm is
/// therefore zero if and only if every component is exactly zero, which is the
/// only row that genuinely has no direction.
fn normalise_row(row: &[f32], index: usize, space: &FaceModel) -> Result<FaceEmbedding> {
  if let Some(component) = row.iter().position(|v| !v.is_finite()) {
    return Err(Error::NonFiniteOutput(NonFiniteOutput::new(
      index, component,
    )));
  }
  let norm = row
    .iter()
    .map(|v| f64::from(*v) * f64::from(*v))
    .sum::<f64>()
    .sqrt();
  if norm == 0.0 {
    return Err(Error::EmbeddingZero(BatchRow::new(index)));
  }
  Ok(FaceEmbedding {
    // Divided in `f64` and narrowed once, at the end: scaling in `f32` would
    // put back the overflow this widening exists to remove.
    values: row.iter().map(|v| (f64::from(*v) / norm) as f32).collect(),
    // The one place a space is attached, and it is the embedder's OWN
    // manifest — the artifact these numbers came out of — never a value a
    // caller states about them.
    space: *space,
  })
}

#[cfg(test)]
mod tests;
