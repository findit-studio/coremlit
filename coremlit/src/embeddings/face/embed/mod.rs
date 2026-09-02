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
//! read off the loaded model's input feature — after the load contract below
//! has established that the feature admits exactly one shape, so it is the
//! graph's ONLY batch and not the default a flexible one would also report
//! ([`FaceEmbedder::batch_capacity`]) — and the slice is chunked to it, so a
//! batch-1 export and a batch-8 export are the same call site.
//!
//! That capacity is the ARTIFACT's number and nothing bounds it, so the buffers
//! it sizes are the one place this door's arithmetic is over a value it did not
//! choose. Both per-prediction element counts are `checked_mul`'d at load and
//! carried (`TensorElements`), and EVERY buffer sized from an artifact- or
//! manifest-controlled number is reserved fallibly: the two per-prediction
//! tensors through `zeroed_tensor`, the per-row embedding through
//! `embedding_buffer`, and the de-aliasing gather `Features::from_provider`
//! may run through `MultiArray::deep_copy`. A wrap and an abort are the two
//! ways an accepted model used to end the caller's process instead of
//! returning an error, and a fallible reservation on some of the buffers is
//! not a fix for either — the abort happens at whichever one is still
//! infallible.
//!
//! # The load contract is a value, and a type proves it was checked
//!
//! [`FaceEmbedder`] holds a `Checked` model, never a bare [`Model`]: the only
//! constructor of that wrapper takes this door's `LoadContract` and runs it,
//! so removing the load check is a compile error rather than a mutation that
//! survives every test. The contract is BUILT at load rather than written down
//! as a constant, because two of its numbers are not this module's — the
//! embedding width is the caller's [`FaceModel::dim`], and the batch is the
//! artifact's, read back off the checked model.
//!
//! Three declarations used to load clean here and then fail, or degrade, at
//! predict time. Each is a clause now:
//!
//! - a graph declaring the manifest's input **plus another REQUIRED input**,
//!   which [`FaceEmbedder::embed`] never sends;
//! - a graph declaring an **`MLState` buffer**, which is not an input at all —
//!   it lives in its own dictionary, so a stateful graph naming exactly these
//!   two features cleared every check this door used to make;
//! - a **flexible** input whose DEFAULT shape reads `[n, 3, 112, 112]`.
//!   [`crate::FeatureInfo::shape`] reports the default of a `RangeDim` or
//!   enumerated feature rather than a bound, so its numbers are
//!   indistinguishable from a pinned graph's and the batch read off it would be
//!   a default rather than a fact.
//!
//! ## A legacy `neuralNetwork` export is refused at load, deliberately
//!
//! An earlier version of this door accepted an EMPTY declared shape on either
//! feature — the legacy `neuralnetwork` specification leaves shapes undeclared
//! — by guessing a batch-one graph and leaving the guess to be caught at
//! predict time. The guess is gone: a feature this door cannot read a rank off
//! is refused when the model is loaded, and so is one whose geometry is not
//! [`crate::ShapeConstraint::Fixed`].
//!
//! The refusal is wider than the empty shape it started from, and that is
//! worth stating rather than discovering. [`crate::ShapeConstraint`]'s measured
//! table records that **every output of a `neuralnetwork` export reports
//! `Unspecified`, even when its input is fixed** — so no artifact in that
//! format loads here, whatever it declares. Fail-closed is the choice: a shape
//! this door guesses is a shape nothing measured. If a real legacy artifact
//! ever matters it arrives as a contract variant with a measurement behind it,
//! not as an arm with a guess in it.

use std::path::Path;

use crate::{
  ComputeUnits, DataType, Model, ModelDescription, MultiArray,
  embeddings::face::{
    align::{AlignedFace, TEMPLATE_BYTES, TEMPLATE_SIZE},
    artifact::{ArtifactDigest, digest_around},
    error::{
      AllocationFailed, BatchRow, ContractMismatch, ElementCountOverflow, EmbeddingSpaceField,
      Error, IncomparableEmbeddings, NonFiniteOutput, NonFinitePreprocessing, OutputElementCount,
      OutputShape, PredictionTensor, PreprocessingField, PreprocessingMap, Result,
      ZeroEmbeddingWidth,
    },
  },
  model::contract::{
    Checked, ContractViolation, Dim, FeatureContract, LoadContract, StateContract,
  },
};

/// Elements one face occupies in the input tensor: `3 · 112 · 112`.
///
/// Numerically [`TEMPLATE_BYTES`] — [`write_row`] maps every byte of the RGB8
/// template to exactly one tensor element — and spelled as that constant rather
/// than re-multiplied, so the row stride [`FaceEmbedder::build_input`] slices by
/// cannot drift from the template it is slicing.
const FACE_ELEMENTS: usize = TEMPLATE_BYTES;

/// The channel order a model's input tensor expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum ChannelOrder {
  /// Red, green, blue — the order [`AlignedFace`] stores.
  Rgb,
  /// Blue, green, red — OpenCV's order, and AdaFace's original checkpoints'.
  Bgr,
}

/// The axis order a model's input tensor expects.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "kebab-case"))]
pub enum TensorLayout {
  /// `[batch, channel, height, width]` — the PyTorch/ONNX convention every
  /// ArcFace export in the census uses.
  Nchw,
  /// `[batch, height, width, channel]` — the TensorFlow convention.
  Nhwc,
}

/// One `f32` preprocessing field as its SEMANTIC identity: the bit pattern,
/// with the representations that mean the same thing folded onto one.
///
/// Two foldings, and each closes a case where a raw `to_bits` comparison says
/// "different" about preprocessing that is not:
///
/// - **`−0.0` and `+0.0` are one value.** They are the same real number, so
///   `byte · scale + bias` is the same function either way — and the produced
///   TENSOR agrees, because [`write_row`] normalises every zero it writes to
///   `+0.0`. That parenthetical used to read the other way: the tensor *could*
///   differ in the sign of a zero (with a negative scale, byte `0` gives
///   `−0.0 + +0.0 = +0.0` against `−0.0 + −0.0 = −0.0`) and "nothing
///   downstream reads the sign of a zero" was the argument for tolerating it.
///   A graph can read it — `sign`, `copysign`, and `1/x` as `+∞` against `−∞`
///   — so one space had two tensors. The producer canonicalises now, and this
///   fold is the whole truth rather than half of it.
/// - **Every NaN is one value.** This serves [`Preprocessing`]'s own [`Eq`]
///   lawfulness and nothing else. The type is public and both its constructors
///   are `const`, so a NaN `Preprocessing` can be built, and without the fold
///   it would not equal itself. It does not serve a broken manifest reaching a
///   comparison: [`FaceEmbedder::load`] refuses a preprocessing whose map does
///   not stay in `f32` ([`Error::NonFinitePreprocessing`]) — the two fields
///   AND the map they make, at both ends of the byte range — so no stamped
///   [`EmbeddingSpace`] carries a NaN.
///
/// Both foldings are reflexive, symmetric and transitive, which is what lets
/// [`Preprocessing`] and [`EmbeddingSpace`] be [`Eq`] at all — `f32`'s own
/// `PartialEq` is not an equivalence relation.
fn canonical_bits(value: f32) -> u32 {
  if value.is_nan() {
    f32::NAN.to_bits()
  } else if value == 0.0 {
    0
  } else {
    value.to_bits()
  }
}

/// One model's host-side preprocessing: `value = byte · scale + bias[channel]`.
///
/// `scale` and `bias` are in the MODEL's channel order, so a BGR model's
/// per-channel bias is written blue-first. Both forms in the module table
/// reduce to this: a divisor `d` and a mean `m` are `scale = 1/d`,
/// `bias = −m/d`.
///
/// Equality is `canonical_bits` on the two float fields rather than `f32`'s
/// own `==` or a raw bit comparison, which is why this is [`Eq`] and [`Hash`].
/// It is the SAME relation [`EmbeddingSpace`] decides a cosine by — one type
/// must not carry two equalities that disagree.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Preprocessing {
  order: ChannelOrder,
  layout: TensorLayout,
  scale: f32,
  bias: [f32; 3],
}

impl PartialEq for Preprocessing {
  #[inline]
  fn eq(&self, other: &Self) -> bool {
    self.order == other.order
      && self.layout == other.layout
      && canonical_bits(self.scale) == canonical_bits(other.scale)
      && self.bias.map(canonical_bits) == other.bias.map(canonical_bits)
  }
}

impl Eq for Preprocessing {}

impl core::hash::Hash for Preprocessing {
  #[inline]
  fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
    core::hash::Hash::hash(&self.order, state);
    core::hash::Hash::hash(&self.layout, state);
    core::hash::Hash::hash(&canonical_bits(self.scale), state);
    core::hash::Hash::hash(&self.bias.map(canonical_bits), state);
  }
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

/// The space one embedder's vectors live in: everything that decides what the
/// NUMBERS are.
///
/// # Which fields, and why each one is here
///
/// - **`artifact`** — the [`ArtifactDigest`] of the bytes
///   [`FaceEmbedder::load`] read. The trained parameters ARE most of the
///   function that produced a vector, and every other field is schema two
///   unrelated exports are free to agree on.
/// - **`output`** — the feature the tensor was read from. For a graph with two
///   `[batch, dim]` heads the output name selects *which function produced the
///   numbers*, so it is not routing.
/// - **`input`** — the feature the pixels were written to, for the same
///   reason on the other side.
/// - **`dim`** and **`preprocessing`** — the width, and the pixels-to-tensor
///   map the host applied before inference.
///
/// A previous round removed the two names as "IO routing" and stated the
/// remaining hole — "two distinct artifacts with one schema are one space" —
/// as a residual. Both halves of that were the same mistake one level apart,
/// and `artifact` closes it: **two `FaceEmbedding`s compare only if
/// byte-identical artifacts produced them, read from the same output feature,
/// fed through the same input feature, with the same host preprocessing.**
///
/// # Produced, never assembled
///
/// There is no public constructor. The only value of this type a caller can
/// obtain came from [`FaceEmbedder::space`] or off a [`FaceEmbedding`], and in
/// both cases it is the space an embedder this crate loaded actually ran in.
/// A caller still chooses which artifact to load and what preprocessing to
/// declare — but not what the loaded bytes hash to.
#[derive(Debug, Clone, Copy)]
pub struct EmbeddingSpace {
  artifact: ArtifactDigest,
  input: &'static str,
  output: &'static str,
  dim: usize,
  preprocessing: Preprocessing,
}

impl EmbeddingSpace {
  /// The space a loaded artifact and its manifest name together.
  ///
  /// **The one place the projection happens**, so "which fields are the space"
  /// has a single answer with a single definition — and so a unit gate builds
  /// a space exactly the way [`FaceEmbedder::load`] builds one, rather than
  /// through a second spelling that could drift from it.
  #[inline]
  const fn of(artifact: ArtifactDigest, manifest: &FaceModel) -> Self {
    Self {
      artifact,
      input: manifest.input,
      output: manifest.output,
      dim: manifest.dim,
      preprocessing: manifest.preprocessing,
    }
  }

  /// The SHA-256 identity of the artifact's bytes.
  #[inline]
  pub const fn artifact(&self) -> ArtifactDigest {
    self.artifact
  }

  /// The input feature the pixels were written to.
  #[inline]
  pub const fn input(&self) -> &'static str {
    self.input
  }

  /// The output feature the embedding was read from.
  #[inline]
  pub const fn output(&self) -> &'static str {
    self.output
  }

  /// The embedding width — [`FaceModel::dim`], reconciled against the
  /// artifact's declared output at load.
  #[inline]
  pub const fn dim(&self) -> usize {
    self.dim
  }

  /// The host-side preprocessing the pixels went through.
  #[inline]
  pub const fn preprocessing(&self) -> Preprocessing {
    self.preprocessing
  }
}

impl PartialEq for EmbeddingSpace {
  /// **Defined as `space_difference` finding nothing** — the very walk
  /// [`FaceEmbedding::dot`] refuses on, so `a == b` and `a.dot(b)` cannot
  /// disagree about whether two vectors are comparable. One relation, written
  /// once.
  #[inline]
  fn eq(&self, other: &Self) -> bool {
    space_difference(*self, *other).is_none()
  }
}

impl Eq for EmbeddingSpace {}

impl core::hash::Hash for EmbeddingSpace {
  #[inline]
  fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
    core::hash::Hash::hash(&self.artifact, state);
    core::hash::Hash::hash(&self.input, state);
    core::hash::Hash::hash(&self.output, state);
    core::hash::Hash::hash(&self.dim, state);
    core::hash::Hash::hash(&self.preprocessing, state);
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
///
/// Equality here is equality of all four fields, with the floats compared by
/// `canonical_bits` as [`Preprocessing`] compares them — the SAME relation
/// [`EmbeddingSpace`] decides a cosine by, on the four fields the two types
/// share. It is deliberately a different type's relation nonetheless, because
/// the two types answer different questions: a manifest is what a caller
/// declares about an artifact, and a space is what an embedder actually ran
/// in. The space carries a fifth field a manifest cannot know — the
/// [`ArtifactDigest`] of the bytes [`FaceEmbedder::load`] read — so two equal
/// manifests name one space only when one artifact produced both.
/// `manifest_equality_and_space_identity_are_one_relation` pins that they
/// never disagree on the four they share.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
  ///
  /// **A `dim` of zero is accepted HERE and refused at
  /// [`FaceEmbedder::load`]**, with [`Error::ZeroEmbeddingWidth`]. This
  /// constructor is `const` and total, so it cannot refuse anything; the refusal
  /// is at the one place that can, and it is a load-time refusal rather than a
  /// clause of the contract because the contract a zero width builds is
  /// satisfiable — see that error for the walk.
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
/// measurement. So each embedding carries the [`EmbeddingSpace`] it was
/// produced in and [`Self::dot`] REFUSES a pair that disagrees.
///
/// **A value here is produced, never assembled.** There is no public
/// constructor for a `FaceEmbedding`: the only way to obtain one is
/// [`FaceEmbedder::embed`], which stamps the space of its own bound manifest
/// onto every row it returns.
///
/// # What that guarantee IS, and what two previous rounds claimed it was
///
/// It was written down here that "there is no public `FaceModel` constructor,
/// so a caller-built `FaceModel` can never be stamped on a vector". **That was
/// false**: [`FaceModel::new`] and [`Preprocessing::new`] are both public
/// `const fn`. It was then written down that every field of the space is
/// therefore a value the caller chose — **and that is false now**, which is
/// the point of [`EmbeddingSpace::artifact`]. Both sentences are kept here
/// inverted rather than deleted, because the shape of the mistake is the
/// useful part: a guarantee is worth what its unforgeable half is worth, and
/// twice the unforgeable half was named wrongly.
///
/// What is actually true, stated as narrowly as it holds:
///
/// - a `FaceEmbedding`'s components came out of a real prediction by an
///   embedder this crate loaded — no caller can assemble one;
/// - the space stamped on it is the space that embedder actually ran in, not a
///   claim made later at the comparison site;
/// - its `artifact` is the SHA-256 of the bytes [`FaceEmbedder::load`] read.
///   A caller chooses which artifact to load; they do not choose what it
///   hashes to, and neither [`crate::embeddings::face::ArtifactDigest`] nor
///   [`EmbeddingSpace`] has a public constructor;
/// - `dim` was reconciled against the artifact's own declared output width at
///   load, with no exception left: the legacy `neuralNetwork` form that used to
///   declare no shape and carry the caller's claim to the first prediction is
///   now refused at load (see the module doc);
/// - the preprocessing half is caller-stated and cannot be otherwise: the
///   artifact does not declare its own normalisation, and the preprocessing
///   really is what the host did to the pixels, so comparing it is sound.
///
/// The strongest claim the type can now make, and it makes it: **two
/// `FaceEmbedding`s compare only if they were produced by byte-identical
/// artifacts, read from the same output feature, fed through the same input
/// feature, with the same host preprocessing. [`Self::dot`] cannot return a
/// score across different weights.**
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
/// call site, nothing is *resolved* at the comparison site, and the part this
/// crate cannot refute is stated below instead of claimed away.
///
/// **Not taken: a minted, process-unique token, and the reason is a
/// difference in the two problems rather than a lighter standard.** A cohort is
/// one object, so minting inside it costs nothing. A face embedding space is
/// legitimately produced by MORE than one producer — `&self` inference means
/// fan-out is one [`FaceEmbedder`] per worker over the same artifact (see that
/// type's doc), and a per-load token would refuse the cross-worker comparisons
/// those workers exist to make. **The digest sidesteps that because it is an
/// identity of the BYTES rather than of the load**: same bundle, same value,
/// on every worker and every machine. There is also no lookup here to remove:
/// `calibrate`'s defect was a question that could be answered twice
/// differently, and [`Self::dot`] asks nothing of caller-owned state — it
/// compares two values' own recorded spaces.
///
/// # The residuals, stated
///
/// Three, and each is a different kind of thing:
///
/// - **a numerically identical re-export is REFUSED.** One set of weights
///   written out twice — recompiled, or renamed — is two artifacts by digest,
///   so their embeddings do not compare and a caller who re-exports has to
///   re-embed. That is loud and correct under this crate's provenance model,
///   where `MODELS_LOCK` already treats bundle bytes as identity: two files
///   that are not the same bytes are not the same artifact. It is a real cost,
///   and it is the price of the guarantee above rather than an oversight;
/// - **a caller who states the wrong preprocessing gets a consistent,
///   off-distribution space.** That is misuse rather than conflation — the
///   vectors are all wrong the same way, so they still compare with each other
///   — and it is unclosable here, because the artifact declares no
///   normalisation for this crate to check the claim against. The one part of
///   the claim that does not need the artifact IS checked: a preprocessing
///   whose map `byte ↦ byte · scale + bias` leaves `f32` is refused at load
///   ([`Error::NonFinitePreprocessing`]), evaluated at both ends of the byte
///   range rather than on the two fields alone, so no stamped space carries a
///   NaN;
/// - **[`AlignedFace::from_template_pixels`] keeps its documented hole on the
///   pixel side.** Bring-your-own-alignment cannot be checked: pixels aligned
///   to some other template, or not aligned at all, pass that constructor and
///   degrade every cosine silently. See its own doc.
///
/// What is NOT a residual any more is the one the round before last recorded
/// here: "two distinct artifacts with one schema are one space as far as this
/// type can see". They are two spaces, and
/// `two_artifacts_with_one_schema_are_two_spaces` is the gate.
#[derive(Debug, Clone, PartialEq)]
pub struct FaceEmbedding {
  /// The unit-norm components. Always `space.dim()` of them: the only
  /// constructor fills this from a row the space's own width cut.
  ///
  /// **A `Vec` rather than the `Box<[f32]>` this was, because there is no
  /// fallible way to reach the boxed slice.** The width is the manifest's and
  /// the buffer is reserved with `try_reserve_exact` (see
  /// [`embedding_buffer`]); `Vec::into_boxed_slice` then documents that it
  /// "discards excess capacity like `shrink_to_fit`", and `try_reserve_exact`
  /// documents that the allocator "may give the collection more space than it
  /// requests" — so the conversion is a REALLOCATION the standard library is
  /// free to perform, and its failure mode is `handle_alloc_error`, the abort
  /// this whole path exists to remove. Keeping the `Vec` is what makes the
  /// fallible reservation the last allocation on the path. Nothing public
  /// changes: the field is private and every accessor reads it as a slice.
  values: Vec<f32>,
  /// The space of the embedder that produced this vector.
  space: EmbeddingSpace,
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
  ///
  /// **Not part of the fallibly-reserved class, deliberately.** Every buffer
  /// `embed` sizes from the artifact's batch or the manifest's width is
  /// reserved through `try_reserve_exact` because the door ACCEPTED a model
  /// whose numbers it did not choose and must not then abort. This is the
  /// other side of that: the vector already exists, so duplicating it asks the
  /// allocator for a length it has just served, and the same is true of this
  /// type's derived [`Clone`]. [`Self::as_slice`] borrows the same components
  /// and allocates nothing, for a caller that does not need the copy.
  #[inline]
  pub fn to_vec(&self) -> Vec<f32> {
    self.values.to_vec()
  }

  /// The [`EmbeddingSpace`] this vector lives in — the space of the embedder
  /// that produced it.
  ///
  /// Readable so a caller storing embeddings can group them, and so a stored
  /// vector can be checked against a freshly loaded embedder before a batch of
  /// comparisons rather than one at a time.
  ///
  /// **Reading it forges nothing, and neither does stating one.** The four
  /// manifest fields are values the caller handed to [`FaceEmbedder::load`];
  /// the fifth is the digest of the bytes that door read, which the caller
  /// does not choose. And an [`EmbeddingSpace`] cannot be assembled at all —
  /// nor can a [`FaceEmbedding`]. See this type's doc for what that does and
  /// does not establish.
  #[inline]
  pub const fn space(&self) -> EmbeddingSpace {
    self.space
  }

  /// The dot product with `other`, which for two unit vectors in one space is
  /// their cosine.
  ///
  /// # Errors
  /// [`Error::IncomparableEmbeddings`] if the two came from different spaces,
  /// naming the first field that differs.
  ///
  /// **Fallible rather than a sentinel.** This used to return `0.0` for a
  /// width mismatch, which is also what a measured orthogonal pair returns —
  /// so a caller could not tell an incompatible model migration from a face
  /// that did not match. A width mismatch is now one arm of the space check,
  /// reported as [`EmbeddingSpaceField::Dim`], and the arm nothing could
  /// report before — equal widths, different spaces — is the rest of it.
  ///
  /// # Accumulated in `f64`, clamped, narrowed once
  ///
  /// A `f32` accumulation returns scores a cosine cannot have. The width-10
  /// witness in `a_unit_vector_never_scores_above_one_against_itself` scored
  /// `1.0000001192` against ITSELF — one `f32` ulp above one, which makes
  /// `acos` NaN, a `1 − cos` distance negative, and a threshold sweep produce
  /// a bucket that should be empty.
  ///
  /// **The clamp is correct here, and an error would be wrong.** Every stored
  /// component is the `f32` rounding of an exact unit component, so the sum is
  /// `Σ uᵢvᵢ(1 + εᵢ)(1 + δᵢ)` with `|εᵢ|, |δᵢ| ≤ 2⁻²⁴`: it can exceed one by
  /// at most `(1 + 2⁻²⁴)² − 1 = 1.19e-7` (measured worst case `8.8e-8`), and
  /// that excess is NARROWING ERROR rather than anything about the two faces.
  /// There is nothing to report, so the value is put back inside the interval
  /// its type promises instead of being turned into a refusal.
  #[inline]
  pub fn dot(&self, other: &Self) -> Result<f32> {
    if let Some(field) = space_difference(self.space, other.space) {
      return Err(Error::IncomparableEmbeddings(IncomparableEmbeddings::new(
        field,
      )));
    }
    let sum: f64 = self
      .values
      .iter()
      .zip(other.values.iter())
      .map(|(x, y)| f64::from(*x) * f64::from(*y))
      .sum();
    Ok(sum.clamp(-1.0, 1.0) as f32)
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

/// The first field of two spaces that puts their embeddings in different
/// spaces, or `None` when they are one space.
///
/// **The single definition of that relation.** [`EmbeddingSpace`]'s
/// [`PartialEq`] is this function, so the equality a caller can test and the
/// refusal [`FaceEmbedding::dot`] raises cannot disagree; there is no second
/// walk to drift from this one.
///
/// The `f32`s are compared by [`canonical_bits`] — neither `f32`'s `==`, which
/// makes a NaN scale unequal to itself, nor a raw `to_bits`, which makes `−0.0`
/// a different space from `+0.0`. Both of those are relations that answer a
/// question about the SPELLING where the question asked is about the function.
///
/// Every field is compared, not short-circuited — the array is built before
/// `find_map` walks it — so the order decides only WHICH field is named when
/// several differ at once. [`EmbeddingSpaceField::Artifact`] leads because
/// when the WEIGHTS differ every other agreement is coincidence: reporting a
/// matching width or a matching divisor would be true and would point a reader
/// at the wrong thing.
fn space_difference(left: EmbeddingSpace, right: EmbeddingSpace) -> Option<EmbeddingSpaceField> {
  let (lp, rp) = (left.preprocessing(), right.preprocessing());
  let bias = |b: [f32; 3]| b.map(canonical_bits);
  [
    (
      EmbeddingSpaceField::Artifact,
      left.artifact() != right.artifact(),
    ),
    (
      EmbeddingSpaceField::InputFeature,
      left.input() != right.input(),
    ),
    (
      EmbeddingSpaceField::OutputFeature,
      left.output() != right.output(),
    ),
    (EmbeddingSpaceField::Dim, left.dim() != right.dim()),
    (EmbeddingSpaceField::ChannelOrder, lp.order() != rp.order()),
    (
      EmbeddingSpaceField::TensorLayout,
      lp.layout() != rp.layout(),
    ),
    (
      EmbeddingSpaceField::PreprocessingScale,
      canonical_bits(lp.scale()) != canonical_bits(rp.scale()),
    ),
    (
      EmbeddingSpaceField::PreprocessingBias,
      bias(lp.bias()) != bias(rp.bias()),
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
  /// A [`Checked`], never a bare [`Model`]: [`load_contract`] builds the only
  /// contract this door states and [`Checked::new`] is the only way a model is
  /// wrapped in one, so deleting the check from [`Self::load`] does not
  /// compile.
  model: Checked,
  manifest: FaceModel,
  /// The space every vector this embedder produces is stamped with, built at
  /// load from the manifest AND the digest of the bytes that were read.
  space: EmbeddingSpace,
  /// The graph's own batch dimension AND declared rank. The rank is what
  /// decided the contract; the batch is READ BACK off the checked model, which
  /// is what makes [`Dim::AnyFixed`] a fact rather than a claim. The rank is
  /// carried, not just the capacity: a model that declares the unbatched
  /// rank-3 form has to be fed a rank-3 tensor.
  input: InputContract,
  /// The output form the graph declared, so a predicted tensor is checked
  /// against the axes it promised and not merely against an element count.
  output: OutputContract,
  /// The element counts one prediction allocates, established at load with
  /// `checked_mul` and carried so no inference-time site multiplies the
  /// artifact's batch again.
  elements: TensorElements,
}

impl FaceEmbedder {
  /// Loads a compiled `.mlmodelc` and binds it to `manifest`.
  ///
  /// The manifest's feature names and embedding width are reconciled against
  /// the model's declared contract here, so a manifest that names the wrong
  /// features, or claims the wrong width, fails at load rather than producing
  /// a plausible-looking wrong vector.
  ///
  /// # The contract, and where each of its numbers comes from
  ///
  /// The model is checked against a crate-internal `LoadContract` and held as
  /// a `Checked` whose only constructor runs that check, so there is no
  /// separate list of validations here to fall out of step with what the door
  /// needs:
  ///
  /// ```text
  /// input   manifest.input()   f32  [n, 3, 112, 112]  n AnyFixed, the rest Exactly
  ///                            f32  [3, 112, 112]     the unbatched form
  ///                     NHWC:  f32  [n, 112, 112, 3] / [112, 112, 3]
  /// output  manifest.output()  f32  [n, dim]          n Exactly the input's batch
  ///                            f32  [dim]             only where that batch is 1
  /// state   none
  /// ```
  ///
  /// `dim` is [`FaceModel::dim`] and the layout is
  /// [`Preprocessing::layout`] — both the caller's. `n` is the ARTIFACT's: the
  /// declared RANK of the input feature picks which of the two forms the
  /// contract states, and the batch axis is an "any one fixed size" axis —
  /// this door does not require a batch, it reads back whichever one the graph
  /// pins. It reads it off the CHECKED model, so the number
  /// [`Self::batch_capacity`] reports came from a description established to
  /// admit exactly one shape, rather than from the default a flexible graph
  /// also reports.
  ///
  /// **A batch the door reads is a batch the door has to size two buffers
  /// from, and that is checked here rather than trusted.** `n` is the
  /// artifact's, so nothing bounds it: `n = usize::MAX / 1000` is a well-formed
  /// pinned shape that used to load clean and then wrap `n · 112 · 112 · 3` on
  /// the way to an allocation, panicking on the first row slice out of the
  /// too-short buffer that resulted. `TensorElements` computes both counts
  /// with `checked_mul` at load, refuses on overflow
  /// ([`Error::ElementCountOverflow`]) and is CARRIED, so `embed` never
  /// multiplies the artifact's batch again. There is no cap — see that type for
  /// why a proof is the right shape and a cap is not.
  ///
  /// The OUTPUT's batch axis is `Exactly` that same number rather than
  /// `AnyFixed`, because this door does more than read it: [`Self::embed`]
  /// sends `n` faces and cuts `n` rows out of what comes back, so a graph that
  /// takes `n` and emits some other row count is one this door cannot use, and
  /// refusing it at load is the difference between a mismatch and a batch of
  /// silently wrong vectors.
  ///
  /// # What is refused, and why a list of feature checks was not enough
  ///
  /// The contract is complete over the three members of
  /// [`crate::ModelDescription`] that can make an otherwise-conformant
  /// prediction fail, not just over the two features this door names: a graph
  /// carrying the manifest's input plus another REQUIRED input clears every
  /// per-feature clause and then fails every prediction, and a STATE buffer is
  /// not an input at all — it lives in its own dictionary, so a stateful graph
  /// declaring exactly these two features clears the input set too and only
  /// then meets [`Self::embed`], which predicts through the stateless API
  /// CoreML does not let a stateful model be called with.
  ///
  /// Both features must be `float32` MULTI-ARRAYS with a PINNED shape.
  /// Inference supplies and extracts nothing else, so an f16 export — or an
  /// `ImageType` feature, which carries no shape and no element type at all
  /// and is what both third-party CoreML ArcFace builds this module's doc
  /// surveys declare — is refused here rather than loading clean and failing
  /// every prediction. A FLEXIBLE feature is refused for a different reason,
  /// and the module doc carries it along with the deliberate refusal of every
  /// legacy `neuralNetwork` export.
  ///
  /// # The digest of the loaded bytes BRACKETS the load
  ///
  /// [`Self::space`] — and therefore every [`FaceEmbedding`] this embedder
  /// produces — carries the [`ArtifactDigest`] of the directory this path
  /// names. That is what makes the space an identity of the WEIGHTS rather
  /// than of a schema, and it is computed here because this is the only place
  /// that knows both the bytes and the manifest. Same bundle ⇒ same digest,
  /// on every worker and every machine, so the cross-worker comparisons
  /// `&self` inference exists to allow are unaffected.
  ///
  /// **The digest is taken twice: before the artifact is opened and again
  /// after the description has been read.** A load and a hash are two separate
  /// walks of a path this crate does not own, and this doc used to record the
  /// gap between them as unclosable and out of scope. It is neither. A bundle
  /// replaced in that window was loaded as A and stamped as B, so every vector
  /// carried an identity belonging to weights that never ran — which is
  /// precisely the confusion the digest exists against, arriving through the
  /// digest. `digest_around` brackets the whole read and refuses on a
  /// mismatch with [`Error::ArtifactChangedDuringLoad`]; see that payload for
  /// the A→B→A residual this does NOT close, and why a private snapshot was
  /// declined.
  ///
  /// The old ordering — hash last, after the cheap rejections — is gone with
  /// it, and the cost is stated rather than hidden: a manifest that does not
  /// match the artifact now pays ONE walk of the bundle instead of none.
  /// A read cannot be bracketed by a hash that starts after it.
  ///
  /// # Errors
  /// [`Error::NonFinitePreprocessing`] if the manifest's map
  /// `byte ↦ byte · scale + bias` does not stay in `f32` — either field, or
  /// the map itself at an end of the byte range, which two finite fields can
  /// still fail; [`Error::ZeroEmbeddingWidth`] if the manifest's width is zero,
  /// which is refused at the manifest because no contract clause can refuse it
  /// and the failure is otherwise a PANIC in [`Self::embed`]'s row split;
  /// [`Error::Load`] if CoreML rejects the model;
  /// [`Error::ContractMismatch`]
  /// if the model declares no feature by the manifest's name, if the declared
  /// rank of either feature is one no contract of this door's can be built
  /// from (an undeclared shape included), or if a named feature's element
  /// type, rank, shape flexibility or any one axis is not the contract's;
  /// [`Error::UnsatisfiableInput`] if it requires an input this door never
  /// sends; [`Error::UnsatisfiableState`] if it declares a state buffer;
  /// [`Error::ElementCountOverflow`] if the batch the graph pins makes either
  /// the input or the output tensor's element count leave `usize`;
  /// [`Error::ArtifactDigest`] if the artifact's bytes cannot be read — which
  /// fails the load rather than producing vectors with no identity;
  /// [`Error::ArtifactChangedDuringLoad`] if the bundle does not hash the same
  /// before and after the load, so which bytes CoreML read is not known.
  pub fn load(
    model_path: impl AsRef<Path>,
    manifest: FaceModel,
    options: FaceEmbedderOptions,
  ) -> Result<Self> {
    let model_path = model_path.as_ref();
    // Hash, read, hash again. Everything this function does with the path is
    // inside the bracket, so a bundle replaced while it is in flight is
    // refused rather than loaded as A and stamped as B.
    let ((model, input, resolved), artifact) = digest_around(model_path, || {
      let model = Model::load(model_path, options.compute())?;
      let resolved = load_contract(model.description(), &manifest)?;
      let model = Checked::new(model, &resolved.contract).map_err(contract_violation)?;
      let input = InputContract::read_back(model.description(), manifest.input(), resolved.rank);
      Ok((model, input, resolved))
    })?;
    let space = EmbeddingSpace::of(artifact, &manifest);
    Ok(Self {
      model,
      manifest,
      space,
      input,
      output: resolved.output,
      elements: resolved.elements,
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

  /// The [`EmbeddingSpace`] every vector from this embedder is stamped with.
  ///
  /// **The only public producer of a space**, and the reason it is here rather
  /// than on [`FaceModel`]: half of a space is the manifest and the other half
  /// is the digest of the bytes that were loaded, which a manifest does not
  /// know. Read it to group stored embeddings, or to check a stored vector
  /// against a freshly loaded embedder once instead of on every comparison.
  #[inline]
  pub const fn space(&self) -> EmbeddingSpace {
    self.space
  }

  /// The graph's own batch dimension, read off its input feature at load —
  /// after the load contract established that the feature admits exactly one
  /// shape, so this is the graph's only batch rather than its default one.
  ///
  /// [`Self::embed`] chunks any slice to this, so it is a throughput fact
  /// rather than a call-site constraint.
  ///
  /// **Not bounded, and it does not need to be.** No cap is imposed on what a
  /// graph may pin here; what [`Self::load`] establishes instead is that the
  /// two element counts this number sizes fit `usize` (`TensorElements`), and
  /// what `zeroed_tensor` establishes is that a buffer the allocator will not
  /// give is an error rather than an abort. A batch too large to be useful is
  /// the artifact's business; a batch that makes this door misbehave is not
  /// loadable.
  #[inline]
  pub const fn batch_capacity(&self) -> usize {
    self.input.batch
  }

  /// The embedding width — [`FaceModel::dim`], reconciled against the model at
  /// load, and never zero: [`Self::load`] refuses a zero-width manifest.
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
  /// [`Error::AllocationFailed`] if a buffer the graph's batch or the
  /// manifest's width sizes cannot be allocated — either per-prediction tensor,
  /// or any one of the per-row embeddings a chunk is cut into — an error rather
  /// than an abort, which is the whole reason every one of them is reserved
  /// fallibly; [`Error::Tensor`] / [`Error::Prediction`] on a tensor or CoreML
  /// failure, which includes
  /// [`PredictionError::AliasCopyFailed`](crate::PredictionError::AliasCopyFailed)
  /// carrying [`TensorError::AllocationFailed`](crate::TensorError::AllocationFailed)
  /// when a graph echoes its input back as its output and the de-aliasing copy
  /// cannot be allocated;
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
  ///
  /// **Every buffer here is reserved fallibly, and the peak is why that has to
  /// include the per-row one.** The flat gather buffer is `elements.output`
  /// long, and the `chunk.len()` rows it is then cut into are each `dim` long
  /// and all live at once, so this function's high-water mark is `batch · dim`
  /// TWICE over on the Rust side, beside both native tensors. Reserving the
  /// flat buffer fallibly and the rows infallibly would move the abort rather
  /// than remove it — the rows are the larger half once the chunk is full.
  ///
  /// # Panics
  /// Never, and the one that could is `chunks_exact(dim)`, which panics on a
  /// chunk size of zero. `dim` is the MANIFEST's, so nothing about the graph
  /// bounds it; [`load_contract`] refuses a zero-width manifest before this
  /// door exists, which is what makes the split total.
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
    check_predicted_shape(
      features.shape(),
      features.count(),
      self.output,
      batch,
      dim,
      self.elements.output,
    )?;

    let mut flat = zeroed_tensor(PredictionTensor::Output, self.elements.output)?;
    features.copy_into::<f32>(&mut flat)?;
    let mut rows = Vec::with_capacity(chunk.len());
    let space = self.space;
    for (offset, row) in flat.chunks_exact(dim).take(chunk.len()).enumerate() {
      rows.push(normalise_row(row, first_row + offset, space)?);
    }
    Ok(rows)
  }

  /// Builds the `[batch, …]` input tensor for one chunk, zero-padding the tail
  /// rows a short chunk leaves.
  ///
  /// The length is the one [`TensorElements`] proved at load, not
  /// `batch · 3 · pixels` recomputed here, and it is reserved through
  /// [`zeroed_tensor`] so an allocation the artifact's batch makes impossible
  /// is an error rather than an abort.
  fn build_input(&self, chunk: &[AlignedFace]) -> Result<MultiArray> {
    let preprocessing = self.manifest.preprocessing();
    let mut data = zeroed_tensor(PredictionTensor::Input, self.elements.input)?;
    for (row, face) in chunk.iter().enumerate() {
      write_row(
        &mut data[row * FACE_ELEMENTS..(row + 1) * FACE_ELEMENTS],
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
  /// The contract the door runs on, taken off a model that has ALREADY been
  /// checked against the [`LoadContract`] `rank` came from.
  ///
  /// # Why the batch is read here and not kept from the declaration
  ///
  /// [`Dim::AnyFixed`] is specified as an axis whose value the door reads back
  /// after the check, and the two moments are not the same fact. Before it,
  /// [`crate::FeatureInfo::shape`] can be the DEFAULT shape of a flexible
  /// feature — a `RangeDim` or enumerated graph reports one it will accept
  /// others beside. After it, the feature is
  /// [`crate::ShapeConstraint::Fixed`], which is what an `AnyFixed` axis
  /// requires, so the number is the graph's only batch rather than a reading of
  /// its declaration. [`load_contract`] therefore returns the RANK and not the
  /// batch: the rank is what the contract is built from, the batch is what the
  /// door then runs on.
  ///
  /// # Panics
  /// Never, for a description [`Checked::new`] accepted against that contract:
  /// the check established that `feature` is declared and has exactly this
  /// rank.
  fn read_back(description: &ModelDescription, feature: &str, rank: InputRank) -> Self {
    let batch = match rank {
      InputRank::Unbatched => 1,
      InputRank::Batched => description
        .input(feature)
        .and_then(|declared| declared.shape().first().copied())
        .expect("the load contract established this feature and its rank"),
    };
    Self { batch, rank }
  }
}

/// The element counts one prediction allocates, PROVED at load to fit `usize`.
///
/// # Why a proof, and why it is kept rather than recomputed
///
/// Both counts are products with the ARTIFACT's batch in them, and that batch
/// is not a number this crate or its caller chose: the input contract states
/// the batch axis as [`Dim::AnyFixed`] and [`InputContract::read_back`] reads
/// back whatever the graph pins, and nothing in a `.mlmodelc` bounds it. A
/// `usize::MAX / 1000` batch declares a perfectly well-formed fixed shape.
///
/// Computed with `*` at the point of use, `batch · 112 · 112 · 3` then wraps
/// silently in a release build; `build_input` allocates the wrapped length and
/// panics on the first `row * FACE_ELEMENTS ..` slice, so a model the door
/// ACCEPTED terminates the caller. Computed here with `checked_mul` it is
/// [`Error::ElementCountOverflow`] — a refusal at load, from a value the door
/// then carries, so no inference-time site multiplies an artifact-derived
/// number again and there is no second spelling to drift.
///
/// **No cap.** A cap would be an enumeration of how big is too big; the product
/// either fits `usize` or it does not. Fitting `usize` is also strictly weaker
/// than the memory existing, which is why the buffers themselves are still
/// reserved fallibly — see [`zeroed_tensor`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TensorElements {
  /// `batch · 112 · 112 · 3` — what [`FaceEmbedder::build_input`] allocates.
  input: usize,
  /// `batch · dim` — what [`FaceEmbedder::predict_chunk`] allocates, and the
  /// count [`check_predicted_shape`] measures the predicted tensor against.
  output: usize,
}

impl TensorElements {
  /// Both counts for a graph of this batch under a manifest of this width.
  ///
  /// # Errors
  /// [`Error::ElementCountOverflow`] naming the tensor whose product leaves
  /// `usize`, the batch, and that tensor's per-row count. The input is checked
  /// first only so that a description overflowing both reports one thing; each
  /// is refused on its own, and `dim` is the manifest's, so the output count
  /// can overflow where the input's does not.
  fn of(batch: usize, dim: usize) -> Result<Self> {
    let input = batch
      .checked_mul(FACE_ELEMENTS)
      .ok_or(Error::ElementCountOverflow(ElementCountOverflow::new(
        PredictionTensor::Input,
        batch,
        FACE_ELEMENTS,
      )))?;
    let output =
      batch
        .checked_mul(dim)
        .ok_or(Error::ElementCountOverflow(ElementCountOverflow::new(
          PredictionTensor::Output,
          batch,
          dim,
        )))?;
    Ok(Self { input, output })
  }
}

/// A zeroed `f32` buffer of `elements`, reserved FALLIBLY.
///
/// `vec![0.0f32; n]` on an artifact-derived `n` answers "the allocator will not
/// give me that" by aborting the process, which is not a thing a caller can
/// handle and not a thing a `Result`-returning door should do.
/// [`TensorElements`] has already proved the count fits `usize`, and that is a
/// strictly weaker fact than the memory existing — a batch of `2⁵⁵` counts fine
/// and asks for petabytes — so the reservation is `try_reserve_exact` and the
/// failure is [`Error::AllocationFailed`], naming the tensor and the length
/// that was refused.
///
/// The `resize` cannot allocate: the reservation above it already secured
/// capacity for exactly `elements`.
///
/// This covers the two PER-PREDICTION tensors only. The per-ROW buffer
/// [`normalise_row`] fills has the same provenance and the same failure mode,
/// and is reserved through [`embedding_buffer`].
fn zeroed_tensor(tensor: PredictionTensor, elements: usize) -> Result<Vec<f32>> {
  let mut data: Vec<f32> = Vec::new();
  data
    .try_reserve_exact(elements)
    .map_err(|_| Error::AllocationFailed(AllocationFailed::new(tensor, elements)))?;
  data.resize(elements, 0.0);
  Ok(data)
}

/// What [`load_contract`] resolves off a description: the contract
/// [`Checked::new`] then checks, and the three facts the door runs on once it
/// has passed.
struct Resolved {
  /// The contract to check the description against.
  contract: LoadContract,
  /// The rank the graph declared, and therefore the rank it must be fed.
  rank: InputRank,
  /// The output form a predicted tensor is later measured against.
  output: OutputContract,
  /// The two element counts one prediction allocates.
  elements: TensorElements,
}

/// The load contract this door states for `manifest`, with the two forms
/// [`FaceEmbedder`] carries: the RANK it must feed, and the output form a
/// predicted tensor is later measured against.
///
/// **Pure over a [`ModelDescription`], so every clause is drivable with no
/// model present.** [`FaceEmbedder::load`] runs exactly this and then
/// [`Checked::new`], which is [`crate::model::contract::check_load_contract`]
/// over the same description; this module's fixtures run that same pair. It is
/// the seam that lets a door staging no artifact still gate its own load path.
///
/// # Why the description is read BEFORE it is checked
///
/// A contract cannot be checked before it exists, and this one's SHAPE comes
/// off the artifact: the declared rank of each feature decides which of the two
/// forms the contract states. That reading is not trusted — it is what the
/// check then confirms. A declaration that lies about its geometry, such as a
/// flexible input whose default shape reads exactly `[n, 3, 112, 112]`, builds
/// a contract it then fails; the reading never becomes a fact without passing.
///
/// The order the clauses are checked in does not change that verdict, only
/// which clause is named: the batch this reads off the input is used to state
/// the OUTPUT's row count, and if the input's own declaration is not pinned,
/// the input clause refuses the model whether it is consulted first or last.
///
/// # The manifest's own numbers are refused before the graph is read
///
/// A manifest carries exactly two kinds of number, and both are checked here
/// rather than where they are used:
///
///   - **`preprocessing`'s `scale` and `bias`** — refused when the affine MAP
///     they make leaves `f32`, which also covers the degenerate constructions
///     that reach them: [`Preprocessing::from_mean_and_divisor`] divides by its
///     `divisor`, and a zero one makes `scale` `±inf` while a NaN one makes it
///     NaN. Either way this clause is what stops it, and it stops it on the
///     resulting `scale` rather than on the divisor, so there is no second
///     predicate to drift. A `scale` of zero is NOT degenerate: it writes a
///     constant tensor, which is well defined if useless, and an output row
///     that then comes back all-zero meets [`Error::EmbeddingZero`] like any
///     other.
///   - **`dim`** — refused at zero, see below.
///
/// Nothing else the door multiplies, divides or chunks by is the manifest's.
/// `TEMPLATE_SIZE` and [`FACE_ELEMENTS`] are compile-time constants, and the
/// only other number in the arithmetic is the ARTIFACT's batch, whose zero
/// [`input_form`] refuses and whose overflow [`TensorElements`] does.
///
/// # Errors
/// [`Error::NonFinitePreprocessing`] for a manifest whose map leaves `f32`;
/// [`Error::ZeroEmbeddingWidth`] for a manifest of zero width, which no clause
/// of the contract it would otherwise build can refuse — see
/// [`ZeroEmbeddingWidth`];
/// [`Error::ContractMismatch`] naming the feature whose declaration no contract
/// of this door's can be built from: absent, or of a rank that is neither the
/// batched nor the unbatched form — an undeclared (empty) shape included, and a
/// declared batch of zero with it;
/// [`Error::ElementCountOverflow`] if the batch the input feature declares
/// makes either tensor's element count leave `usize` — see [`TensorElements`].
fn load_contract(description: &ModelDescription, manifest: &FaceModel) -> Result<Resolved> {
  // Before anything about the graph: a manifest whose MAP does not stay in
  // `f32` makes elements of the input tensor non-finite, and the manifest is
  // copied verbatim into the `EmbeddingSpace` stamped on the vectors. Refusing
  // it here is what keeps a NaN out of a produced space, and therefore what
  // makes `canonical_bits`' NaN fold a statement about `Preprocessing`'s own
  // `Eq` and nothing more. The check is on the map at both ends of the byte
  // range, not on the two fields alone — `f32::MAX` and `0.0` are both finite
  // and their map is not.
  if let Some(field) = non_finite_preprocessing(manifest.preprocessing()) {
    return Err(Error::NonFinitePreprocessing(NonFinitePreprocessing::new(
      field,
    )));
  }
  // Also before the graph, and before any contract is built: a manifest of ZERO
  // width. This is the door's only refusal of a manifest NUMBER, and it has to
  // live here because nothing after it can make it — every clause a zero width
  // reaches is satisfied by it, and the failure lands in `predict_chunk`'s
  // `chunks_exact(dim)` as a panic. See [`ZeroEmbeddingWidth`] for the walk.
  if manifest.dim() == 0 {
    return Err(Error::ZeroEmbeddingWidth(ZeroEmbeddingWidth::new(
      manifest.output(),
    )));
  }
  let layout = manifest.preprocessing().layout();
  let declared_input = description.input(manifest.input()).ok_or_else(|| {
    Error::ContractMismatch(ContractMismatch::new(
      manifest.input().to_string(),
      "a declared input feature".to_string(),
      format!("inputs {:?}", feature_names(description.inputs())),
    ))
  })?;
  let (rank, batch) = input_form(declared_input.shape()).ok_or_else(|| {
    Error::ContractMismatch(ContractMismatch::new(
      manifest.input().to_string(),
      format!(
        "{layout:?} shaped [n, 3, {TEMPLATE_SIZE}, {TEMPLATE_SIZE}] (or without the batch axis)"
      ),
      format!("{:?}", declared_input.shape()),
    ))
  })?;

  // The batch is now a number, and both tensors are sized from it. Prove the
  // two products fit `usize` HERE, where the batch first becomes one, rather
  // than at the two allocation sites where a wrap would be silent.
  let elements = TensorElements::of(batch, manifest.dim())?;

  let declared_output = description.output(manifest.output()).ok_or_else(|| {
    Error::ContractMismatch(ContractMismatch::new(
      manifest.output().to_string(),
      "a declared output feature".to_string(),
      format!("outputs {:?}", feature_names(description.outputs())),
    ))
  })?;
  let form = output_form(declared_output.shape(), batch).ok_or_else(|| {
    Error::ContractMismatch(ContractMismatch::new(
      manifest.output().to_string(),
      format!(
        "shaped [{batch}, {}] (or [{}] for a batch-one graph)",
        manifest.dim(),
        manifest.dim()
      ),
      format!("{:?}", declared_output.shape()),
    ))
  })?;

  let contract = LoadContract::new(
    vec![FeatureContract::new(
      manifest.input(),
      DataType::F32,
      input_dims(rank, layout),
    )],
    vec![FeatureContract::new(
      manifest.output(),
      DataType::F32,
      output_dims(form, batch, manifest.dim()),
    )],
    StateContract::None,
  );
  Ok(Resolved {
    contract,
    rank,
    output: form,
    elements,
  })
}

/// The first thing about `preprocessing` that does not stay in `f32` — the
/// MAP, evaluated at both ends of the byte range, attributed to a field where
/// a field is what is wrong.
///
/// # The map, not the fields — and the endpoints are a PROOF
///
/// Checking `scale` and `bias` for finiteness one at a time was an enumeration
/// of what can go wrong that missed the thing the fields are for:
/// `scale = f32::MAX` with `bias = 0` is two perfectly finite numbers whose
/// map writes `+inf` for every byte from 2 upwards, so the input tensor was
/// non-finite from a manifest the load had blessed.
///
/// `byte ↦ byte · scale + bias[channel]` is **affine in `byte`**. The exact
/// value at any byte therefore lies between the exact values at `0` and `255`,
/// and rounding to nearest is monotone — so if both endpoints round to finite
/// `f32`, every byte between them does. That is a proof over the whole domain,
/// not a sample of it, which is why two evaluations per channel are enough and
/// why no third byte needs a rule of its own. NaN cannot arise at all once
/// `scale` and `bias` are finite.
///
/// The expression evaluated here is [`write_row`]'s own `mul_add`, so what is
/// proved is what will be written rather than a differently associated
/// stand-in for it.
///
/// # Which end fires, and which end cannot
///
/// Only the far one, and the reason is arithmetic rather than a gap in the
/// gate: `byte 0 · scale + bias` is exactly `bias` for any finite `scale`, so
/// once the field checks have passed, the byte-0 endpoint is finite by
/// construction. It is evaluated because the PAIR is what proves the 254 bytes
/// between them, and the gate says so where it pins the mutation that drops
/// the far one.
///
/// # The FIRST, not all of them
///
/// One thing that is definitely wrong is more actionable than a list assembled
/// to look thorough, and the same rule [`space_difference`] follows. The
/// failure is attributed to the most specific thing about it: a non-finite
/// `scale` is [`PreprocessingField::Scale`], a non-finite `bias` is
/// [`PreprocessingField::Bias`], and a map that leaves `f32` out of two finite
/// fields is [`PreprocessingField::Map`] naming the channel and the endpoint.
fn non_finite_preprocessing(preprocessing: Preprocessing) -> Option<PreprocessingField> {
  let scale = preprocessing.scale();
  for (channel, offset) in preprocessing.bias().iter().enumerate() {
    for byte in [u8::MIN, u8::MAX] {
      if f32::from(byte).mul_add(scale, *offset).is_finite() {
        continue;
      }
      if !scale.is_finite() {
        return Some(PreprocessingField::Scale);
      }
      if !offset.is_finite() {
        return Some(PreprocessingField::Bias(channel));
      }
      return Some(PreprocessingField::Map(PreprocessingMap::new(
        channel, byte,
      )));
    }
  }
  None
}

/// Map a [`ContractViolation`] into this module's error vocabulary.
///
/// **A newtype variant over the violation itself would be the house shape, and
/// it is not available here.** [`ContractViolation`] is `pub(crate)`, so a
/// public [`Error`] variant carrying one would export a private type; widening
/// the whole contract vocabulary to `pub` for one door's error message is a
/// larger change to a shared type than this door's convenience earns. So the
/// violation is RENDERED, the way `audio::identity` renders it: the four
/// per-feature clauses land in [`Error::ContractMismatch`], which already
/// carries a feature name and an expected/actual pair, and the two
/// "unsatisfiable" clauses keep newtype variants of their own, because they are
/// about what the door cannot SUPPLY rather than about a feature's shape.
fn contract_violation(violation: ContractViolation) -> Error {
  let (feature, expected, actual) = match violation {
    ContractViolation::UnsatisfiableInput(input) => {
      return Error::UnsatisfiableInput(input.name().to_string());
    }
    ContractViolation::UnsatisfiableState(state) => {
      return Error::UnsatisfiableState(state.name().to_string());
    }
    ContractViolation::Missing(missing) => (
      missing.feature(),
      "a declared feature".to_string(),
      "missing".to_string(),
    ),
    ContractViolation::DataType(mismatch) => {
      (mismatch.feature(), mismatch.expected(), mismatch.observed())
    }
    ContractViolation::Rank(mismatch) => {
      (mismatch.feature(), mismatch.expected(), mismatch.observed())
    }
    ContractViolation::Flexibility(mismatch) => {
      (mismatch.feature(), mismatch.expected(), mismatch.observed())
    }
    ContractViolation::Axis(mismatch) => {
      (mismatch.feature(), mismatch.expected(), mismatch.observed())
    }
  };
  Error::ContractMismatch(ContractMismatch::new(feature.to_string(), expected, actual))
}

/// Which of the two forms a model's input feature declares, and the batch that
/// form implies — or `None` for a declared rank that is neither.
///
/// **The RANK is all this decides**, and that is the narrowing this door's
/// adoption of the load contract bought. The element type, and that the three
/// trailing axes really are a `3 × 112 × 112` template face in the manifest's
/// layout, are clauses of the contract [`load_contract`] then builds — checked
/// once by [`Checked::new`] rather than twice in two spellings that could
/// disagree.
///
/// An EMPTY shape is refused, and that is deliberate rather than an oversight:
/// the legacy `neuralnetwork` specification declares none, and this used to
/// resolve one to a batch-one guess. The module doc carries the argument.
///
/// A declared batch of ZERO is refused here rather than by the contract,
/// because the contract cannot express it: [`Dim::AnyFixed`] asks only that the
/// axis admit exactly one size, and zero is one size. [`FaceEmbedder::embed`]
/// would divide its work into chunks of zero and never terminate, so it is a
/// contract mismatch and not a capacity.
fn input_form(shape: &[usize]) -> Option<(InputRank, usize)> {
  match shape.len() {
    3 => Some((InputRank::Unbatched, 1)),
    4 if shape[0] > 0 => Some((InputRank::Batched, shape[0])),
    _ => None,
  }
}

/// The contract axes for an input feature of this rank and layout — the
/// per-axis form of the shape [`input_shape`] later builds, so the geometry the
/// door STATES and the geometry it FEEDS come from one pair of arms.
fn input_dims(rank: InputRank, layout: TensorLayout) -> Vec<Dim> {
  let face = match layout {
    TensorLayout::Nchw => [
      Dim::Exactly(3),
      Dim::Exactly(TEMPLATE_SIZE),
      Dim::Exactly(TEMPLATE_SIZE),
    ],
    TensorLayout::Nhwc => [
      Dim::Exactly(TEMPLATE_SIZE),
      Dim::Exactly(TEMPLATE_SIZE),
      Dim::Exactly(3),
    ],
  };
  match rank {
    InputRank::Unbatched => face.to_vec(),
    InputRank::Batched => {
      let mut dims = Vec::with_capacity(1 + face.len());
      // Not `Exactly`: this door does not require a batch, it reads back
      // whichever one the graph pins.
      dims.push(Dim::AnyFixed);
      dims.extend_from_slice(&face);
      dims
    }
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
}

/// Which output form a model's output feature declares, or `None` for a
/// declared rank that is neither.
///
/// A `[dim]` shape is a batch-one form. Declared against a batch of 4 it is not
/// a shorthand, it is a contradiction, so it is refused here rather than
/// allowed to build a contract the predicted tensor could never satisfy. An
/// EMPTY shape is refused for the reason [`input_form`] refuses one — this is
/// where the `Undeclared` arm used to be.
fn output_form(shape: &[usize], batch: usize) -> Option<OutputContract> {
  match shape.len() {
    1 if batch == 1 => Some(OutputContract::Flat),
    2 => Some(OutputContract::Batched),
    _ => None,
  }
}

/// The contract axes for the output feature: `[batch, dim]`, or the bare
/// `[dim]` a batch-one graph may declare instead.
///
/// The batch axis is [`Dim::Exactly`] rather than [`Dim::AnyFixed`] because
/// this door does more than read it back — see [`FaceEmbedder::load`].
fn output_dims(form: OutputContract, batch: usize, dim: usize) -> Vec<Dim> {
  match form {
    OutputContract::Flat => vec![Dim::Exactly(dim)],
    OutputContract::Batched => vec![Dim::Exactly(batch), Dim::Exactly(dim)],
  }
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
///
/// `elements` is that required count — [`TensorElements::output`], proved to
/// fit `usize` at load — and is PASSED rather than recomputed as `batch · dim`,
/// which is the same artifact-derived product that would wrap here as silently
/// as it would at an allocation.
///
/// This is a DIFFERENT moment from the load contract, which is why it survives
/// the adoption of [`Checked`]: the contract established what the graph
/// DECLARES, and this measures what one prediction actually produced.
fn check_predicted_shape(
  shape: &[usize],
  count: usize,
  contract: OutputContract,
  batch: usize,
  dim: usize,
  elements: usize,
) -> Result<()> {
  let batched = [batch, dim];
  let flat = [dim];
  let expected: &[usize] = match contract {
    OutputContract::Batched => &batched,
    OutputContract::Flat => &flat,
  };
  if shape != expected {
    return Err(Error::OutputShape(OutputShape::new(
      shape.to_vec(),
      expected.to_vec(),
    )));
  }
  if count != elements {
    return Err(Error::OutputElementCount(OutputElementCount::new(
      count, elements,
    )));
  }
  Ok(())
}

/// Writes one aligned face into `row` as `3 · 112 · 112` preprocessed floats.
///
/// **Every zero written is `+0.0`.** [`Preprocessing`]'s equality folds `±0`
/// onto one value, so two manifests differing only in the sign of a zero bias
/// are ONE [`EmbeddingSpace`] and their embeddings compare — but `byte · scale
/// + bias` does not fold it: pixel `0` with `scale = −1` gives `−0.0` from the
/// multiply, and `+0.0` and `−0.0` as the bias then write two different bit
/// patterns. A graph can read that difference (`sign`, `copysign`, and `1/x`
/// as `+∞` against `−∞`), so one space could produce two tensors. It is
/// canonicalised here, at the producer, rather than compared away downstream:
/// see `a_written_zero_is_positive_zero_whichever_sign_the_bias_carries`.
///
/// **Every value written is finite, and that is the LOAD's guarantee rather
/// than this function's.** `non_finite_preprocessing` evaluates the `mul_add`
/// below at byte `0` and byte `255` for each channel, and the map is affine in
/// `byte`, so the two endpoints being finite proves all 256 are. A check here
/// would add nothing the load has not already established, and one per pixel
/// would pay 37 632 branches to re-derive it.
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
      // `+ 0.0` normalises `−0.0` to `+0.0` and leaves every other value
      // alone: under round-to-nearest the sum of two zeros of opposite sign
      // is `+0.0`, and `x + 0.0` is exactly `x` for any nonzero finite `x`
      // (and for `±∞`). One add, no branch, and the sign of a zero cannot
      // reach the tensor.
      let value = f32::from(source[pixel * 3 + source_channel]).mul_add(scale, *offset) + 0.0;
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
///
/// # Errors
/// [`Error::NonFiniteOutput`] naming the row and the first non-finite
/// component; [`Error::EmbeddingZero`] naming a row whose (finite) components
/// are all exactly zero; [`Error::AllocationFailed`] if the row's own buffer —
/// `dim` `f32`s, the manifest's width, reserved through [`embedding_buffer`]
/// — is one the allocator will not serve.
fn normalise_row(row: &[f32], index: usize, space: EmbeddingSpace) -> Result<FaceEmbedding> {
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
  let mut values = embedding_buffer(row.len())?;
  // `extend` cannot allocate: `Vec::reserve` is documented to do nothing when
  // the capacity is already sufficient, and the reservation above secured
  // exactly `row.len()`.
  //
  // Divided in `f64` and narrowed once, at the end: scaling in `f32` would put
  // back the overflow this widening exists to remove.
  values.extend(row.iter().map(|v| (f64::from(*v) / norm) as f32));
  Ok(FaceEmbedding {
    values,
    // The one place a space is attached, and it is the space of the embedder
    // that just ran — the function these numbers actually came out of — rather
    // than one stated about them afterwards at a comparison site.
    space,
  })
}

/// A buffer for ONE normalised output row, reserved FALLIBLY.
///
/// The sibling of [`zeroed_tensor`], and the reason that one was not the whole
/// class. `zeroed_tensor` covers the two PER-PREDICTION buffers;
/// [`normalise_row`] allocates once PER ROW, and it used to do it with a
/// `collect` into a `Box<[f32]>` — which for a `TrustedLen` iterator is
/// `Vec::with_capacity(row.len())` under the covers, so `handle_alloc_error`
/// and an abort when the allocator refuses.
///
/// The width is the MANIFEST's `dim`, the same number `elements.output` is
/// half of, so the same "fits `usize`, may not fit memory" gap applies; and
/// this one MULTIPLIES. Across a chunk the rows duplicate the whole output
/// tensor while the flat gather buffer and both native tensors are still live,
/// so the peak is `batch · dim` twice over — which is exactly the regime where
/// an artifact large enough to matter would abort a caller AFTER the fallibly
/// reserved flat buffer had succeeded.
///
/// Returns an EMPTY `Vec` with the capacity secured, rather than a filled one:
/// [`normalise_row`] then `extend`s it, which cannot allocate because
/// `Vec::reserve` is documented to do nothing when the capacity already
/// suffices.
fn embedding_buffer(elements: usize) -> Result<Vec<f32>> {
  let mut values: Vec<f32> = Vec::new();
  values.try_reserve_exact(elements).map_err(|_| {
    Error::AllocationFailed(AllocationFailed::new(PredictionTensor::Output, elements))
  })?;
  Ok(values)
}

#[cfg(test)]
mod tests;
