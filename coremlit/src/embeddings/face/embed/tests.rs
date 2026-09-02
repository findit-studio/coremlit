//! Unit gates for the manifest-driven preprocessing, the load contract and the
//! batch plumbing.
//!
//! Almost no model is needed: every function these exercise is pure, and the
//! load-time decision is driven over [`ModelDescription`] fixtures. That is the
//! point of putting preprocessing in a manifest and the load contract in a
//! value — the parts that silently degrade an embedding are testable without an
//! artifact, which matters more here than anywhere else in the crate, because
//! this is the one kit that stages none.
//!
//! Two gates do reach [`FaceEmbedder::load`] itself:
//! `the_face_door_refuses_the_vendored_silero_bundle` runs in every
//! `cargo test`, because that bundle is committed, and
//! `a_load_that_cannot_open_the_artifact_never_walks_it` needs no artifact at
//! all. Between them they pin WHERE the artifact walk sits in the door.

use super::*;
use crate::{
  AxisRange, FeatureInfo, ShapeConstraint,
  embeddings::face::align::TEMPLATE_BYTES,
  model::{RawShapeConstraint, contract::check_load_contract},
};

/// The embedding width every ArcFace-family artifact in issue #115's census
/// emits.
const DIM: usize = 512;

/// A manifest of the given width, standing in for the artifact an embedder
/// would have been loaded against.
fn model(dim: usize) -> FaceModel {
  FaceModel::new("data", "embedding", dim)
}

/// [`model`] with `layout` substituted — the only manifest field the load
/// contract's geometry depends on.
fn manifest(layout: TensorLayout) -> FaceModel {
  let arcface = Preprocessing::ARCFACE;
  model(DIM).with_preprocessing(Preprocessing::new(
    arcface.order(),
    layout,
    arcface.scale(),
    arcface.bias(),
  ))
}

// ── Description fixtures ───────────────────────────────────────────────────
//
// `ModelDescription::from_parts` / `FeatureInfo::from_parts` are `pub(crate)`
// for exactly this: a door that stages NO artifact still gates its whole load
// path, over descriptions CoreML never produced. The shape VERDICT is never
// stated by a fixture — `from_parts` classifies the raw contents — so a fixture
// cannot claim a `Fixed` its own numbers do not support.

/// One multi-array feature, spelled out: the constraint's raw type code, its
/// enumerated shapes, and its per-axis ranges.
fn multi_array(
  name: &str,
  shape: &[usize],
  dtype: DataType,
  optional: bool,
  raw_type: isize,
  enumerated: Vec<Vec<usize>>,
  ranges: Vec<AxisRange>,
) -> FeatureInfo {
  FeatureInfo::from_parts(
    name.to_string(),
    shape.to_vec(),
    Some(dtype),
    optional,
    Some(RawShapeConstraint::new(raw_type, enumerated, ranges)),
  )
}

/// The per-axis ranges a PINNED shape reports.
fn pinned(shape: &[usize]) -> Vec<AxisRange> {
  shape.iter().map(|d| AxisRange::new(*d, 1)).collect()
}

/// A fixed-shape multi-array feature, exactly as a plain coremltools export
/// reports one: raw type 2, its declared shape as the sole enumerated shape,
/// and `(d, 1)` on every axis.
fn fixed(name: &str, shape: &[usize], dtype: DataType) -> FeatureInfo {
  multi_array(
    name,
    shape,
    dtype,
    false,
    2,
    vec![shape.to_vec()],
    pinned(shape),
  )
}

/// A feature as a legacy `neuralnetwork` export declares one: raw type 1, no
/// enumerated shapes, no ranges and an EMPTY shape. Measured on EVERY output of
/// that format, even when its input is fixed — see [`ShapeConstraint`]'s table.
fn undeclared(name: &str) -> FeatureInfo {
  multi_array(name, &[], DataType::F32, false, 1, Vec::new(), Vec::new())
}

/// A conformant artifact's description: one pinned input, one pinned output, no
/// state.
fn graph(input_shape: &[usize], output_shape: &[usize]) -> ModelDescription {
  ModelDescription::from_parts(
    vec![fixed("data", input_shape, DataType::F32)],
    vec![fixed("embedding", output_shape, DataType::F32)],
    Vec::new(),
  )
}

/// The whole load-time decision, run over `description` and mapped into this
/// module's errors — exactly the pair `FaceEmbedder::load` runs between
/// `Model::load` and the digest, with `Checked::new`'s check spelled as the
/// function `Checked::new` itself calls.
fn check(
  description: &ModelDescription,
  manifest: &FaceModel,
) -> Result<(InputContract, OutputContract)> {
  let resolved = load_contract(description, manifest)?;
  check_load_contract(description, &resolved.contract).map_err(contract_violation)?;
  Ok((
    InputContract::read_back(description, manifest.input(), resolved.rank),
    resolved.output,
  ))
}

/// The resolved input contract as a comparable pair, so a gate can assert the
/// declared RANK and not only the numeric capacity. The output half is built to
/// match whatever batch the input declares, so what these rows exercise is the
/// input clause alone.
fn contract_of(shape: &[usize], layout: TensorLayout) -> Option<(usize, InputRank)> {
  let batch = if shape.len() == 4 { shape[0] } else { 1 };
  check(&graph(shape, &[batch, DIM]), &manifest(layout))
    .ok()
    .map(|(input, _)| (input.batch, input.rank))
}

/// A digest standing in for one artifact's bytes.
///
/// `tag` names WHICH artifact, so a gate can say "these two vectors came out
/// of the same weights" or "out of different weights" without staging two
/// `.mlmodelc` directories. The real digests come from `digest_artifact`, and
/// `artifact/tests.rs` is where the hash itself is gated; here only the
/// equality matters.
fn artifact(tag: u8) -> ArtifactDigest {
  ArtifactDigest::from_raw([tag; 32])
}

/// The space `load` would stamp for `manifest`, read out of artifact `tag`.
///
/// Reached through `EmbeddingSpace::of` — the same projection `load` uses — so
/// a gate cannot drift from what the door actually builds.
fn space_of(tag: u8, manifest: &FaceModel) -> EmbeddingSpace {
  EmbeddingSpace::of(artifact(tag), manifest)
}

/// The default space: one artifact, the default manifest, the given width.
fn space(dim: usize) -> EmbeddingSpace {
  space_of(1, &model(dim))
}

/// An [`AlignedFace`] whose byte at `(pixel, channel)` is `pixel + channel`,
/// so a channel swap and a layout swap both change the tensor.
fn ramp_face() -> AlignedFace {
  let mut pixels = vec![0u8; TEMPLATE_BYTES];
  for pixel in 0..TEMPLATE_SIZE * TEMPLATE_SIZE {
    for channel in 0..3 {
      pixels[pixel * 3 + channel] =
        u8::try_from((pixel + channel) % 256).expect("modulo keeps it in range");
    }
  }
  AlignedFace::from_template_pixels(&pixels).expect("exact template length")
}

#[test]
fn arcface_preprocessing_maps_bytes_onto_minus_one_to_one() {
  let p = Preprocessing::ARCFACE;
  assert_eq!(p.order(), ChannelOrder::Rgb);
  assert_eq!(p.layout(), TensorLayout::Nchw);
  // 0 -> -1, 127.5 -> 0, 255 -> +1.
  assert!((0.0f32.mul_add(p.scale(), p.bias()[0]) + 1.0).abs() < 1e-6);
  assert!((255.0f32.mul_add(p.scale(), p.bias()[0]) - 1.0).abs() < 1e-6);
  assert!(127.5f32.mul_add(p.scale(), p.bias()[0]).abs() < 1e-6);
}

#[test]
fn mean_and_divisor_is_the_same_preprocessing_written_the_other_way() {
  // The census states ArcFace as `(x - 127.5) / 127.5`; ARCFACE states it as
  // scale + bias. The two constructors must agree, or the table in the module
  // doc is describing something the code does not do.
  let from_table = Preprocessing::from_mean_and_divisor(
    ChannelOrder::Rgb,
    TensorLayout::Nchw,
    [127.5, 127.5, 127.5],
    127.5,
  );
  assert!((from_table.scale() - Preprocessing::ARCFACE.scale()).abs() < 1e-9);
  for (got, want) in from_table
    .bias()
    .into_iter()
    .zip(Preprocessing::ARCFACE.bias())
  {
    assert!((got - want).abs() < 1e-6);
  }
  // dlib's per-channel form, the row that made `bias` an array rather than a
  // scalar.
  let dlib = Preprocessing::from_mean_and_divisor(
    ChannelOrder::Rgb,
    TensorLayout::Nchw,
    [122.782, 117.001, 104.298],
    256.0,
  );
  assert!((dlib.bias()[0] + 122.782 / 256.0).abs() < 1e-6);
  assert!((dlib.bias()[2] + 104.298 / 256.0).abs() < 1e-6);
  assert!(
    dlib.bias()[0] < dlib.bias()[2],
    "the per-channel means must not collapse"
  );
}

#[test]
fn nchw_rgb_writes_planes_in_channel_order() {
  let face = ramp_face();
  let pixels = TEMPLATE_SIZE * TEMPLATE_SIZE;
  let mut row = vec![0.0f32; 3 * pixels];
  // Identity preprocessing, so the tensor holds the raw bytes and the test is
  // about PLACEMENT rather than arithmetic.
  let identity = Preprocessing::new(ChannelOrder::Rgb, TensorLayout::Nchw, 1.0, [0.0; 3]);
  write_row(&mut row, &face, identity);
  for pixel in [0usize, 1, 17, pixels - 1] {
    for channel in 0..3 {
      let expected = f32::from(face.pixels()[pixel * 3 + channel]);
      assert_eq!(
        row[channel * pixels + pixel],
        expected,
        "NCHW plane {channel}, pixel {pixel}"
      );
    }
  }
}

#[test]
fn bgr_reads_the_opposite_channel_of_the_rgb_template() {
  let face = ramp_face();
  let pixels = TEMPLATE_SIZE * TEMPLATE_SIZE;
  let identity_rgb = Preprocessing::new(ChannelOrder::Rgb, TensorLayout::Nchw, 1.0, [0.0; 3]);
  let identity_bgr = Preprocessing::new(ChannelOrder::Bgr, TensorLayout::Nchw, 1.0, [0.0; 3]);
  let mut rgb = vec![0.0f32; 3 * pixels];
  let mut bgr = vec![0.0f32; 3 * pixels];
  write_row(&mut rgb, &face, identity_rgb);
  write_row(&mut bgr, &face, identity_bgr);
  // Plane 0 of the BGR tensor is plane 2 of the RGB tensor, and vice versa.
  assert_eq!(&bgr[0..pixels], &rgb[2 * pixels..3 * pixels]);
  assert_eq!(&bgr[2 * pixels..3 * pixels], &rgb[0..pixels]);
  assert_eq!(&bgr[pixels..2 * pixels], &rgb[pixels..2 * pixels]);
  assert_ne!(
    &bgr[0..pixels],
    &rgb[0..pixels],
    "the ramp must distinguish the channel orders, or this gate proves nothing"
  );
}

#[test]
fn a_written_zero_is_positive_zero_whichever_sign_the_bias_carries() {
  // `canonical_bits` folds `±0` in the SPACE — two manifests differing only in
  // the sign of a zero bias are one space, and their embeddings compare. The
  // TENSOR did not agree. Pixel `0` with `scale = −1` gives `−0.0` from the
  // multiply, and adding `+0.0` yields `+0.0` while adding `−0.0` yields
  // `−0.0`: two bit patterns from one space, and a graph can tell them apart
  // (`sign`, `copysign`, `1/x` are `+∞` against `−∞`). One relation must not
  // have two answers, so the PRODUCER canonicalises.
  let black =
    AlignedFace::from_template_pixels(&vec![0u8; TEMPLATE_BYTES]).expect("exact template length");
  let signed_zero =
    |bias: f32| Preprocessing::new(ChannelOrder::Rgb, TensorLayout::Nchw, -1.0, [bias; 3]);
  // The premise: the two manifests really are ONE space, so the tensor is the
  // only place the sign could have survived.
  assert_eq!(
    signed_zero(0.0),
    signed_zero(-0.0),
    "the space cannot see the sign of a zero, which is why the tensor must not"
  );

  for bias in [0.0f32, -0.0] {
    let mut row = vec![f32::NAN; 3 * TEMPLATE_SIZE * TEMPLATE_SIZE];
    write_row(&mut row, &black, signed_zero(bias));
    // The BIT PATTERN, not `== 0.0`, which is exactly the comparison that
    // cannot see this.
    for (index, value) in row.iter().enumerate() {
      assert_eq!(
        value.to_bits(),
        0x0000_0000,
        "bias {bias:?} wrote {value:?} (bits {:#010x}) at {index}, not `+0.0`",
        value.to_bits()
      );
    }
  }
}

#[test]
fn a_manifest_whose_preprocessing_is_not_finite_is_refused_at_load() {
  // The other half of the same defect. `canonical_bits` folds every NaN onto
  // one representative so a `Preprocessing` equals itself — the type is public
  // with a public `const` constructor, so a NaN one can be BUILT. What must
  // not happen is that such a manifest reaches an embedder: every value it
  // writes into the input tensor is non-finite, and the space stamped on the
  // vectors carries the NaN forward. The load contract is where the road is
  // cut, so the NaN fold is about the type's algebra and nothing else.
  //
  // Driven through the pure contract path — the same pair `FaceEmbedder::load`
  // runs before it walks the artifact — because this crate stages no face
  // artifact.
  let arcface = Preprocessing::ARCFACE;
  let cases = [
    (f32::NAN, [-1.0f32, -1.0, -1.0], PreprocessingField::Scale),
    (f32::INFINITY, [-1.0, -1.0, -1.0], PreprocessingField::Scale),
    (
      arcface.scale(),
      [-1.0, f32::NAN, -1.0],
      PreprocessingField::Bias(1),
    ),
    (
      arcface.scale(),
      [f32::NEG_INFINITY, -1.0, -1.0],
      PreprocessingField::Bias(0),
    ),
  ];
  for (scale, bias, want) in cases {
    let broken = model(DIM).with_preprocessing(Preprocessing::new(
      arcface.order(),
      arcface.layout(),
      scale,
      bias,
    ));
    let error = check(
      &graph(&[1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], &[1, DIM]),
      &broken,
    )
    .expect_err("a non-finite preprocessing parameter is not loadable");
    assert!(
      matches!(&error, Error::NonFinitePreprocessing(payload) if payload.field() == want),
      "expected NonFinitePreprocessing({want}) for scale={scale:?} bias={bias:?}, got {error:?}"
    );
  }
  // And the finite manifest the same graph is built for still loads, so the
  // refusal is about the parameter and not about the shape.
  assert!(
    check(
      &graph(&[1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], &[1, DIM]),
      &model(DIM)
    )
    .is_ok(),
    "a finite manifest must still load"
  );
}

#[test]
fn a_preprocessing_map_that_leaves_f32_at_a_byte_endpoint_is_refused_at_load() {
  // The gate above refuses each FIELD that is not finite. That was an
  // enumeration of what can go wrong, and it missed the thing the fields are
  // for: `scale = f32::MAX` with `bias = 0` is two perfectly finite numbers
  // whose MAP writes `+inf` for every byte from 2 upwards. So the tensor was
  // non-finite from a manifest the load had blessed, and every claim resting
  // on "no stamped space carries a NaN" rested on the wrong check.
  //
  // The map `byte ↦ byte · scale + bias` is AFFINE in `byte`, so its extremes
  // over `0..=255` are at the two endpoints and rounding is monotone — the two
  // endpoints being finite is therefore a PROOF that all 256 are, not a sample
  // of them. `load` evaluates the exact `mul_add` expression `write_row` uses,
  // at byte 0 and byte 255, for every channel's bias.
  let arcface = Preprocessing::ARCFACE;
  let graph = graph(&[1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], &[1, DIM]);

  // The witness: finite fields, infinite tensor.
  assert!(
    f32::MAX.is_finite() && 0.0f32.is_finite(),
    "both fields are finite, which is why the per-field check admits this"
  );
  assert!(
    !f32::from(255u8).mul_add(f32::MAX, 0.0).is_finite(),
    "and the map at the far endpoint is not"
  );
  let overflowing = model(DIM).with_preprocessing(Preprocessing::new(
    arcface.order(),
    arcface.layout(),
    f32::MAX,
    [0.0; 3],
  ));
  let error =
    check(&graph, &overflowing).expect_err("a manifest whose map leaves `f32` is not loadable");
  assert!(
    matches!(
      &error,
      Error::NonFinitePreprocessing(payload)
        if payload.field() == PreprocessingField::Map(PreprocessingMap::new(0, u8::MAX))
    ),
    "expected the endpoint that overflows to be named, got {error:?}"
  );
  assert!(
    error.to_string().contains("byte 255"),
    "and a reader must be told WHICH end, got {error}"
  );

  // Per CHANNEL, not per manifest: a bias that only overflows on one channel
  // is named on that channel. `2^127` is finite; `2^127 · 255` is not, so the
  // scale alone already overflows — use a scale that is fine on its own and a
  // bias that pushes one channel over.
  let scale = f32::MAX / 256.0;
  assert!(
    f32::from(255u8).mul_add(scale, 0.0).is_finite(),
    "this scale is safe over the whole byte range on its own"
  );
  let lopsided = model(DIM).with_preprocessing(Preprocessing::new(
    arcface.order(),
    arcface.layout(),
    scale,
    [0.0, 0.0, f32::MAX],
  ));
  let error =
    check(&graph, &lopsided).expect_err("one channel's bias carries that channel out of `f32`");
  assert!(
    matches!(
      &error,
      Error::NonFinitePreprocessing(payload)
        if payload.field() == PreprocessingField::Map(PreprocessingMap::new(2, u8::MAX))
    ),
    "expected channel 2 to be named, got {error:?}"
  );

  // The mutation this pins is "check only byte 0": the witness above overflows
  // at 255 and nowhere near 0. The OTHER half of the proof cannot be pinned
  // the same way and the reason is arithmetic rather than a missing gate —
  // `byte 0 · scale + bias` is exactly `bias` for any finite `scale`, so once
  // the two field checks have passed, the byte-0 endpoint cannot be the one
  // that fires. It is evaluated because the PAIR is what proves the 254 bytes
  // between them, not because either end alone is suspected.
  for (scale, bias) in [(f32::MAX, 0.0f32), (scale, f32::MAX)] {
    assert_eq!(
      f32::from(0u8).mul_add(scale, bias),
      bias,
      "the near endpoint is the bias itself, which the field check has already cleared"
    );
  }

  // And the boundary is exact: the largest scale whose map stays finite loads.
  let largest = f32::MAX / f32::from(255u8);
  assert!(
    f32::from(255u8).mul_add(largest, 0.0).is_finite(),
    "the map at the far endpoint is finite here"
  );
  assert!(
    check(
      &graph,
      &model(DIM).with_preprocessing(Preprocessing::new(
        arcface.order(),
        arcface.layout(),
        largest,
        [0.0; 3],
      ))
    )
    .is_ok(),
    "an extreme but finite map must still load — the check is on the map, not on the magnitude"
  );
}

/// **FALSIFIER (red first).** A manifest of ZERO width used to load, on both
/// output forms, and then panic on the first non-empty `embed`.
///
/// Nothing downstream could refuse it. `FaceModel::new` is `const`, so it
/// cannot; the CONTRACT cannot either — `Dim::Exactly(0)` is a well-formed
/// axis, a `[batch, 0]` feature carrying `(0, 1)` on that axis classifies as
/// `Fixed`, and `TensorElements::of` multiplies to a legitimate `0`. A
/// prediction of no elements then satisfies BOTH of
/// `check_predicted_shape`'s clauses, and `predict_chunk` reaches
/// `flat.chunks_exact(0)`, which panics.
///
/// So the refusal has to be at the producer, and this is it. The empty slice
/// is not the exception it looks like: `embed(&[])` never enters
/// `predict_chunk`, so the panic needs one real face and nothing more.
#[test]
fn a_manifest_of_zero_width_is_refused_at_load() {
  let input = [1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE];

  // The batched output form: `[batch, 0]`.
  let error = check(&graph(&input, &[1, 0]), &model(0))
    .expect_err("a zero-width manifest has no embedding to produce");
  assert!(
    matches!(&error, Error::ZeroEmbeddingWidth(payload) if payload.output() == "embedding"),
    "{error}"
  );
  assert!(error.to_string().contains("embedding"), "{error}");

  // And the flat one a batch-one graph may declare instead: `[0]`. The refusal
  // is the manifest's width, so which form the artifact declares cannot matter
  // — and the width is read before the description is, so a graph declaring
  // some OTHER width is refused by the same clause rather than by a mismatch.
  for output in [vec![0], vec![512]] {
    assert!(
      matches!(
        check(&graph(&input, &output), &model(0)),
        Err(Error::ZeroEmbeddingWidth(_))
      ),
      "output {output:?} must be refused for the manifest's width, not for its own"
    );
  }

  // A width of one still loads, so the refusal is zero and not "small".
  assert!(
    check(&graph(&input, &[1, 1]), &model(1)).is_ok(),
    "a one-wide manifest is degenerate but well defined"
  );
}

#[test]
fn nhwc_interleaves_where_nchw_planes() {
  let face = ramp_face();
  let pixels = TEMPLATE_SIZE * TEMPLATE_SIZE;
  let identity = Preprocessing::new(ChannelOrder::Rgb, TensorLayout::Nhwc, 1.0, [0.0; 3]);
  let mut row = vec![0.0f32; 3 * pixels];
  write_row(&mut row, &face, identity);
  for pixel in [0usize, 5, pixels - 1] {
    for channel in 0..3 {
      assert_eq!(
        row[pixel * 3 + channel],
        f32::from(face.pixels()[pixel * 3 + channel]),
        "NHWC pixel {pixel}, channel {channel}"
      );
    }
  }
}

#[test]
fn preprocessing_is_scale_then_bias_with_the_bias_in_the_models_channel_space() {
  // Added after a mutation SURVIVED: every other preprocessing gate here used
  // an identity preprocessing (scale 1, bias 0), under which
  // `byte · scale + bias` and `(byte + bias) · scale` are the same number. So
  // `write_row`'s composition order was untested, and a swap of it is the
  // classic silent degradation — issue #115's census puts a wrong divisor at
  // `1 − cos ≈ 0.083`, fifty times the ANE's own noise floor, with no error
  // raised anywhere.
  //
  // A BGR order with three DIFFERENT biases pins both halves at once: the
  // composition order, and the fact that `bias` is indexed in the MODEL's
  // channel space while the source byte is fetched from the template's.
  let face = ramp_face();
  let pixels = TEMPLATE_SIZE * TEMPLATE_SIZE;
  let scale = 1.0f32 / 127.5;
  let bias = [-1.0f32, -2.0, -3.0];
  let mut row = vec![0.0f32; 3 * pixels];
  write_row(
    &mut row,
    &face,
    Preprocessing::new(ChannelOrder::Bgr, TensorLayout::Nchw, scale, bias),
  );

  for pixel in [0usize, 128, 255, pixels - 1] {
    for channel in 0..3 {
      // BGR model channel `c` reads template channel `2 - c`.
      let byte = f32::from(face.pixels()[pixel * 3 + (2 - channel)]);
      let expected = byte * scale + bias[channel];
      let got = row[channel * pixels + pixel];
      assert!(
        (got - expected).abs() < 1e-6,
        "plane {channel}, pixel {pixel}: got {got}, expected {expected} (byte {byte} · {scale} + \
         {})",
        bias[channel]
      );
      // The other composition order, spelled out so the gate is visibly
      // discriminating rather than merely arithmetic.
      let swapped = (byte + bias[channel]) * scale;
      assert!(
        (expected - swapped).abs() > 1e-3,
        "this fixture cannot tell scale-then-bias from bias-then-scale, so it proves nothing"
      );
    }
  }
}

/// The declarations real ArcFace exports use, and the contract each resolves
/// to.
///
/// The rank is asserted alongside the capacity: the two rank-3 rows used to
/// resolve to a bare `1`, indistinguishable from the batched `[1, 3, 112, 112]`
/// form, and that discarded bit is what fed them a tensor their graph cannot
/// accept. The batch is now READ BACK off a description the contract check
/// accepted, so each of these numbers is a graph's only batch rather than the
/// default a flexible one would also report.
#[test]
fn the_contract_accepts_the_forms_real_exports_declare() {
  assert_eq!(
    contract_of(&[3, TEMPLATE_SIZE, TEMPLATE_SIZE], TensorLayout::Nchw),
    Some((1, InputRank::Unbatched))
  );
  assert_eq!(
    contract_of(&[1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], TensorLayout::Nchw),
    Some((1, InputRank::Batched))
  );
  assert_eq!(
    contract_of(&[8, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], TensorLayout::Nchw),
    Some((8, InputRank::Batched))
  );
  assert_eq!(
    contract_of(&[TEMPLATE_SIZE, TEMPLATE_SIZE, 3], TensorLayout::Nhwc),
    Some((1, InputRank::Unbatched))
  );
  assert_eq!(
    contract_of(&[4, TEMPLATE_SIZE, TEMPLATE_SIZE, 3], TensorLayout::Nhwc),
    Some((4, InputRank::Batched))
  );
}

/// **The contract itself, as a value, for all three accepted forms.** It is
/// this door's whole statement about an artifact, so it is asserted directly
/// rather than only through what it refuses — a clause that quietly weakens
/// (`AnyFixed` where `Exactly` was meant, or the reverse) changes no refusal
/// this file could otherwise see.
#[test]
fn the_contract_reads_the_batch_and_requires_everything_else() {
  let face_nchw = [
    Dim::Exactly(3),
    Dim::Exactly(TEMPLATE_SIZE),
    Dim::Exactly(TEMPLATE_SIZE),
  ];
  let face_nhwc = [
    Dim::Exactly(TEMPLATE_SIZE),
    Dim::Exactly(TEMPLATE_SIZE),
    Dim::Exactly(3),
  ];

  // Rank-4 NCHW, batch 4: the batch axis is READ (`AnyFixed`), the face axes
  // are REQUIRED, and the output's row count is the input's batch stated as a
  // number rather than read a second time.
  let resolved = load_contract(
    &graph(&[4, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], &[4, DIM]),
    &manifest(TensorLayout::Nchw),
  )
  .expect("a batch-4 NCHW export is one of the accepted forms");
  assert_eq!(
    (resolved.rank, resolved.output),
    (InputRank::Batched, OutputContract::Batched)
  );
  assert_eq!(
    resolved.contract,
    LoadContract::new(
      vec![FeatureContract::new(
        "data",
        DataType::F32,
        [Dim::AnyFixed].into_iter().chain(face_nchw).collect()
      )],
      vec![FeatureContract::new(
        "embedding",
        DataType::F32,
        vec![Dim::Exactly(4), Dim::Exactly(DIM)]
      )],
      StateContract::None,
    )
  );

  // Rank-4 NHWC: the same shape of statement, with the channel axis last.
  let resolved = load_contract(
    &graph(&[2, TEMPLATE_SIZE, TEMPLATE_SIZE, 3], &[2, DIM]),
    &manifest(TensorLayout::Nhwc),
  )
  .expect("a batch-2 NHWC export is one of the accepted forms");
  assert_eq!(
    (resolved.rank, resolved.output),
    (InputRank::Batched, OutputContract::Batched)
  );
  assert_eq!(
    resolved.contract,
    LoadContract::new(
      vec![FeatureContract::new(
        "data",
        DataType::F32,
        [Dim::AnyFixed].into_iter().chain(face_nhwc).collect()
      )],
      vec![FeatureContract::new(
        "embedding",
        DataType::F32,
        vec![Dim::Exactly(2), Dim::Exactly(DIM)]
      )],
      StateContract::None,
    )
  );

  // Rank-3, with the bare `[dim]` output only a batch-one graph can declare:
  // there is no batch axis to read, so the contract has no `AnyFixed` at all.
  let resolved = load_contract(
    &graph(&[3, TEMPLATE_SIZE, TEMPLATE_SIZE], &[DIM]),
    &manifest(TensorLayout::Nchw),
  )
  .expect("the unbatched rank-3 form is one of the accepted forms");
  assert_eq!(
    (resolved.rank, resolved.output),
    (InputRank::Unbatched, OutputContract::Flat)
  );
  assert_eq!(
    resolved.contract,
    LoadContract::new(
      vec![FeatureContract::new(
        "data",
        DataType::F32,
        face_nchw.to_vec()
      )],
      vec![FeatureContract::new(
        "embedding",
        DataType::F32,
        vec![Dim::Exactly(DIM)]
      )],
      StateContract::None,
    )
  );
}

#[test]
fn the_contract_refuses_a_shape_that_is_not_a_template_face() {
  // The layouts must not accept each other's shapes: that swap is exactly the
  // silent-degradation failure the manifest exists to prevent. These three are
  // refused by the contract's per-axis clauses now, not by a hand-written
  // shape match beside them.
  assert_eq!(
    contract_of(&[1, TEMPLATE_SIZE, TEMPLATE_SIZE, 3], TensorLayout::Nchw),
    None
  );
  assert_eq!(
    contract_of(&[1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], TensorLayout::Nhwc),
    None
  );
  assert_eq!(
    contract_of(&[1, 3, 96, TEMPLATE_SIZE], TensorLayout::Nchw),
    None
  );
  // And these three earlier, on a rank no contract of this door's can be built
  // from at all.
  assert_eq!(
    contract_of(&[1, 1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], TensorLayout::Nchw),
    None
  );
  assert_eq!(
    contract_of(&[TEMPLATE_SIZE, TEMPLATE_SIZE], TensorLayout::Nchw),
    None
  );
  // A declared batch of zero would make `embed` chunk by zero. The contract
  // cannot express that — `AnyFixed` asks only for ONE size, and zero is one —
  // so the refusal is the door's own.
  assert_eq!(
    contract_of(&[0, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], TensorLayout::Nchw),
    None
  );
}

// ── The load contract's own clauses ────────────────────────────────────────

/// **A graph carrying the manifest's input plus another REQUIRED input** clears
/// every per-feature clause and then fails on EVERY prediction, because
/// `FaceEmbedder::embed` sends the manifest's input and nothing else.
///
/// This door used to look up the two features it wanted and never ask what else
/// the graph required. A state buffer is not an input; an extra required input
/// is not a feature this door names; neither is visible to a check written per
/// feature, which is why the contract is complete over the description instead.
#[test]
fn the_contract_refuses_an_extra_required_input() {
  let description = ModelDescription::from_parts(
    vec![
      fixed("data", &[4, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], DataType::F32),
      fixed("landmark_hint", &[4, 10], DataType::F32),
    ],
    vec![fixed("embedding", &[4, DIM], DataType::F32)],
    Vec::new(),
  );
  let error = check(&description, &manifest(TensorLayout::Nchw)).unwrap_err();
  assert!(
    matches!(&error, Error::UnsatisfiableInput(name) if name == "landmark_hint"),
    "{error}"
  );
}

/// An OPTIONAL extra input is not that: CoreML runs a prediction that omits
/// one, so it cannot make this door's prediction fail. Optionality is exactly
/// the distinction this needs, and a count of inputs cannot make it.
#[test]
fn the_contract_accepts_an_extra_optional_input() {
  let description = ModelDescription::from_parts(
    vec![
      fixed("data", &[4, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], DataType::F32),
      multi_array(
        "landmark_hint",
        &[4, 10],
        DataType::F32,
        true,
        2,
        vec![vec![4, 10]],
        pinned(&[4, 10]),
      ),
    ],
    vec![fixed("embedding", &[4, DIM], DataType::F32)],
    Vec::new(),
  );
  assert!(check(&description, &manifest(TensorLayout::Nchw)).is_ok());
}

/// **The stateful-graph refusal.** A state buffer is not an ordinary input: it
/// lives in `stateDescriptionsByName`, so a stateful graph declaring exactly
/// these two features clears every per-feature clause AND the input set — and
/// only then meets `FaceEmbedder::embed`, which predicts through the STATELESS
/// API. CoreML requires a stateful model to receive an `MLState` on every
/// prediction, so that either fails or silently throws the persistence away.
#[test]
fn the_contract_refuses_a_graph_that_declares_state() {
  let description = ModelDescription::from_parts(
    vec![fixed(
      "data",
      &[1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE],
      DataType::F32,
    )],
    vec![fixed("embedding", &[1, DIM], DataType::F32)],
    vec![fixed("kv_cache", &[1, 8], DataType::F32)],
  );
  let error = check(&description, &manifest(TensorLayout::Nchw)).unwrap_err();
  assert!(
    matches!(&error, Error::UnsatisfiableState(name) if name == "kv_cache"),
    "{error}"
  );
}

/// **The flexible input that declares this door's exact numbers.**
/// `FeatureInfo::shape` reports the DEFAULT shape of a `RangeDim` input, and an
/// equal-bound `RangeDim` reports `(d, 1)` on every axis too — so the numbers
/// and the per-axis ranges are both indistinguishable from a pinned graph's,
/// and only the whole-feature verdict separates them. It matters twice here: a
/// flexible input is what takes a graph off the accelerator, and this door
/// READS its batch off that shape, so accepting one would make
/// `batch_capacity` a default rather than a fact.
#[test]
fn the_contract_refuses_a_flexible_input_declaring_its_exact_numbers() {
  let shape = [4, 3, TEMPLATE_SIZE, TEMPLATE_SIZE];
  let flexible = multi_array(
    "data",
    &shape,
    DataType::F32,
    false,
    3,
    Vec::new(),
    pinned(&shape),
  );
  assert_eq!(
    flexible.shape_constraint(),
    Some(ShapeConstraint::Range),
    "the fixture must be a flexible feature, not merely a differently spelled fixed one"
  );
  let description = ModelDescription::from_parts(
    vec![flexible],
    vec![fixed("embedding", &[4, DIM], DataType::F32)],
    Vec::new(),
  );
  let error = check(&description, &manifest(TensorLayout::Nchw)).unwrap_err();
  assert!(
    matches!(&error, Error::ContractMismatch(m) if m.feature() == "data"),
    "{error}"
  );
  assert!(error.to_string().contains("range"), "{error}");
}

/// **The arm that used to be `OutputContract::Undeclared`.** A legacy
/// `neuralNetwork` export declares no output shape; this door accepted that,
/// guessed a form and left the guess for the predict-time check. It is refused
/// at load now.
///
/// The refusal is wider than this one fixture and that is deliberate: measured
/// in [`ShapeConstraint`]'s table, EVERY output of a `neuralnetwork` export
/// reports `Unspecified` even when its input is fixed, so no artifact in that
/// format loads through this door. The module doc carries the argument.
#[test]
fn the_contract_refuses_an_export_that_declares_no_shape() {
  let output = undeclared("embedding");
  assert_eq!(
    output.shape_constraint(),
    Some(ShapeConstraint::Unspecified),
    "the fixture must be what a `neuralnetwork` output actually reports"
  );
  let description = ModelDescription::from_parts(
    vec![fixed(
      "data",
      &[1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE],
      DataType::F32,
    )],
    vec![output],
    Vec::new(),
  );
  let error = check(&description, &manifest(TensorLayout::Nchw)).unwrap_err();
  assert!(
    matches!(&error, Error::ContractMismatch(m)
      if m.feature() == "embedding" && m.actual() == "[]"),
    "{error}"
  );

  // The same format's INPUT half, for an export that declares neither.
  let description = ModelDescription::from_parts(
    vec![undeclared("data")],
    vec![fixed("embedding", &[1, DIM], DataType::F32)],
    Vec::new(),
  );
  let error = check(&description, &manifest(TensorLayout::Nchw)).unwrap_err();
  assert!(
    matches!(&error, Error::ContractMismatch(m)
      if m.feature() == "data" && m.actual() == "[]"),
    "{error}"
  );
}

/// The output's row count is the INPUT's batch, and the contract states it as a
/// number rather than reading it back: a graph that takes 4 faces and emits 2
/// rows is one `embed` cannot use, and `Dim::AnyFixed` on that axis would let
/// it load and fail on the first prediction instead.
#[test]
fn the_contract_refuses_an_output_whose_batch_is_not_the_inputs() {
  let error = check(
    &graph(&[4, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], &[2, DIM]),
    &manifest(TensorLayout::Nchw),
  )
  .unwrap_err();
  assert!(
    matches!(&error, Error::ContractMismatch(m) if m.feature() == "embedding"),
    "{error}"
  );
}

/// The manifest's width is reconciled against the artifact rather than trusted:
/// a 128-wide graph under a 512-wide manifest is refused, not truncated.
#[test]
fn the_contract_refuses_an_output_of_a_different_width() {
  let error = check(
    &graph(&[4, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], &[4, 128]),
    &manifest(TensorLayout::Nchw),
  )
  .unwrap_err();
  assert!(
    matches!(&error, Error::ContractMismatch(m) if m.feature() == "embedding"),
    "{error}"
  );
}

/// A bare `[dim]` output is a batch-one form. Declared against a batch of 4 it
/// is not a shorthand, it is a contradiction, and accepting it would leave the
/// predicted-tensor check with two incompatible shapes to allow.
#[test]
fn a_flat_output_is_a_batch_one_form_only() {
  let nchw = manifest(TensorLayout::Nchw);
  assert_eq!(
    check(&graph(&[1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], &[DIM]), &nchw)
      .expect("a batch-one graph may declare the bare form")
      .1,
    OutputContract::Flat
  );
  assert!(
    check(&graph(&[4, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], &[DIM]), &nchw).is_err(),
    "[dim] must not resolve against a batch-4 graph"
  );
}

/// A feature the model does not declare at all is refused BY NAME, with the
/// names it does declare in the message.
#[test]
fn the_contract_refuses_a_differently_spelled_feature() {
  let nchw = manifest(TensorLayout::Nchw);
  let description = ModelDescription::from_parts(
    vec![fixed(
      "input_1",
      &[1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE],
      DataType::F32,
    )],
    vec![fixed("embedding", &[1, DIM], DataType::F32)],
    Vec::new(),
  );
  let error = check(&description, &nchw).unwrap_err();
  assert!(
    matches!(&error, Error::ContractMismatch(m)
      if m.feature() == "data" && m.actual() == r#"inputs ["input_1"]"#),
    "{error}"
  );

  let description = ModelDescription::from_parts(
    vec![fixed(
      "data",
      &[1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE],
      DataType::F32,
    )],
    vec![fixed("output_1", &[1, DIM], DataType::F32)],
    Vec::new(),
  );
  let error = check(&description, &nchw).unwrap_err();
  assert!(
    matches!(&error, Error::ContractMismatch(m)
      if m.feature() == "embedding" && m.actual() == r#"outputs ["output_1"]"#),
    "{error}"
  );
}

#[test]
fn normalising_produces_a_unit_vector() {
  let embedding = normalise_row(&[3.0, 4.0], 0, space(2)).expect("finite and nonzero");
  assert_eq!(embedding.dim(), 2);
  assert!((embedding.as_slice()[0] - 0.6).abs() < 1e-6);
  assert!((embedding.as_slice()[1] - 0.8).abs() < 1e-6);
  let norm: f32 = embedding.as_slice().iter().map(|v| v * v).sum();
  assert!((norm - 1.0).abs() < 1e-6);
  assert_eq!(embedding.to_vec(), embedding.as_slice().to_vec());
}

#[test]
fn a_zero_row_is_refused_and_names_the_callers_index() {
  let error = normalise_row(&[0.0, 0.0, 0.0], 7, space(3)).expect_err("zero has no direction");
  assert!(
    matches!(error, Error::EmbeddingZero(payload) if payload.row() == 7),
    "expected EmbeddingZero(7), got {error:?}"
  );
}

#[test]
fn a_non_finite_row_names_the_row_and_the_component() {
  let error =
    normalise_row(&[1.0, f32::NAN, 2.0], 5, space(3)).expect_err("NaN is not an embedding");
  assert!(
    matches!(error, Error::NonFiniteOutput(payload) if payload.row() == 5 && payload.component() == 1),
    "expected NonFiniteOutput(5, 1), got {error:?}"
  );
}

#[test]
fn cosine_is_the_dot_product_of_unit_vectors() {
  let a = normalise_row(&[1.0, 0.0], 0, space(2)).expect("unit");
  let b = normalise_row(&[0.0, 1.0], 0, space(2)).expect("unit");
  let c = normalise_row(&[1.0, 0.0], 0, space(2)).expect("unit");
  assert!(a.cosine(&b).expect("one space").abs() < 1e-6);
  assert!((a.cosine(&c).expect("one space") - 1.0).abs() < 1e-6);
  assert_eq!(
    a.dot(&b).expect("one space"),
    a.cosine(&b).expect("one space")
  );
}

/// A width-10 row whose L2-normalised `f32` components sum their own squares to
/// MORE than one, in both precisions.
///
/// Found by search over small-integer rows, and it has to satisfy two separate
/// conditions at once or it cannot gate what it is here to gate:
///
/// - accumulated the way `dot` used to — sequentially in `f32` — its self-score
///   is `1.0000001192`, one `f32` ulp above one. That is the defect;
/// - accumulated in `f64` it is `1.0000000667703535`, and the excess
///   `6.68e-8` is larger than half an `f32` ulp at one, so it SURVIVES the
///   narrowing back to `f32`. Without that, removing the clamp would change
///   nothing and the clamp would be untested decoration.
const OVER_UNIT_ROW: [f32; 10] = [80.0, 36.0, 30.0, 40.0, 7.0, 12.0, 10.0, 4.0, 75.0, 70.0];

#[test]
fn a_unit_vector_never_scores_above_one_against_itself() {
  // A cosine is a BOUNDED quantity, and every caller is entitled to treat it
  // as one: `acos` of `1.0000001` is NaN, a `1 − cos` distance goes negative,
  // and a threshold sweep gets a bucket that should be empty. `dot` returned
  // `1.0000001192` for a vector against itself.
  //
  // The excess is narrowing error and never a property of the operands. Each
  // stored component is the `f32` rounding of an exact unit component, so
  // `Σ vᵢ²` is `Σ uᵢ²(1 + εᵢ)²` with `|εᵢ| ≤ 2⁻²⁴` — at most
  // `(1 + 2⁻²⁴)² − 1 = 1.19e-7` above one, and never above one for a reason
  // that has anything to do with the two faces. So a clamp is the correct
  // answer here and an error would be the wrong one: there is nothing to
  // report.
  let vector = normalise_row(&OVER_UNIT_ROW, 0, space(10)).expect("finite and nonzero");

  // The witness is only a witness if the excess is really there. Both halves
  // are asserted, so weakening the row reds here rather than silently making
  // the two mutations below survivable.
  let exact: f64 = vector
    .as_slice()
    .iter()
    .map(|v| f64::from(*v) * f64::from(*v))
    .sum();
  assert!(
    exact > 1.0,
    "the witness must overshoot in `f64` too, got {exact:.17}"
  );
  assert!(
    exact as f32 > 1.0,
    "and the overshoot must survive narrowing to `f32`, or the clamp is untestable"
  );

  let self_score = vector.dot(&vector).expect("one vector is in one space");
  assert!(
    self_score <= 1.0,
    "a unit vector scored {self_score:.10} against itself; a cosine that leaves [-1, 1] breaks \
     `acos`, `1 - cos`, and every threshold a caller sets"
  );
  assert!(
    self_score >= 1.0 - 1e-6,
    "and clamping must not cost the answer: {self_score:.10}"
  );

  // The other end, on the same row negated against itself.
  let opposite =
    normalise_row(&OVER_UNIT_ROW.map(|v| -v), 0, space(10)).expect("finite and nonzero");
  let against = vector.dot(&opposite).expect("one space");
  assert!(
    against >= -1.0,
    "an antipodal pair scored {against:.10}, below the floor a cosine has"
  );
}

#[test]
fn one_space_reached_through_two_separately_built_manifests_still_compares() {
  // `&self` inference means fan-out is one embedder per worker over the same
  // artifact, so the SAME space is legitimately produced by more than one
  // producer. A space identity minted per embedder — `calibrate`'s
  // `CalibrationId` shape — would refuse exactly the cross-worker comparisons
  // those workers exist to make. The artifact digest is what sidesteps that:
  // it is an identity of the BYTES, not of the load, so the same bundle read
  // twice is one value. This pins both halves — two separately built manifests
  // over one artifact are one space.
  let a = normalise_row(&[1.0, 0.0], 0, space(2)).expect("unit");
  let b = normalise_row(
    &[1.0, 0.0],
    0,
    space_of(1, &FaceModel::new("data", "embedding", 2)),
  )
  .expect("unit");
  assert_eq!(
    a.cosine(&b).expect("two equal manifests are one space"),
    1.0,
    "two equal manifests must name one space"
  );
}

#[test]
fn embeddings_of_different_widths_are_refused_rather_than_scored_zero() {
  // This used to return `0.0`, which is also what a measured orthogonal pair
  // returns — so a caller could not tell an incompatible model migration from
  // a face that did not match.
  let a = normalise_row(&[1.0, 0.0], 0, space(2)).expect("unit");
  let wide = normalise_row(&[1.0, 0.0, 0.0], 0, space(3)).expect("unit");
  let error = a.cosine(&wide).expect_err("two widths are two spaces");
  assert!(
    matches!(
      error,
      Error::IncomparableEmbeddings(p) if p.field() == EmbeddingSpaceField::Dim
    ),
    "expected IncomparableEmbeddings(Dim), got {error:?}"
  );
}

#[test]
fn embeddings_from_two_preprocessing_spaces_are_refused_not_scored() {
  // The half no width check could ever reach: identical widths, identical
  // components, unrelated spaces. Scored, this pair reads 1.0 — a perfect
  // match between two vectors that mean nothing to each other.
  let rgb = model(2);
  let bgr = rgb.with_preprocessing(Preprocessing::from_mean_and_divisor(
    ChannelOrder::Bgr,
    TensorLayout::Nchw,
    [127.5, 127.5, 127.5],
    127.5,
  ));
  let a = normalise_row(&[1.0, 0.0], 0, space_of(1, &rgb)).expect("unit");
  let b = normalise_row(&[1.0, 0.0], 0, space_of(1, &bgr)).expect("unit");
  assert_eq!(
    a.dot(&a).expect("one space"),
    1.0,
    "the same space must still score"
  );
  let error = a
    .cosine(&b)
    .expect_err("two preprocessing spaces are not one space");
  assert!(
    matches!(
      error,
      Error::IncomparableEmbeddings(p) if p.field() == EmbeddingSpaceField::ChannelOrder
    ),
    "expected IncomparableEmbeddings(ChannelOrder), got {error:?}"
  );

  // Every field of the SPACE decides it, not only the ones a shape check could
  // see. The feature names are deliberately absent — see
  // `feature_names_are_io_routing_and_do_not_decide_the_space`.
  for (want, other) in [
    (
      EmbeddingSpaceField::TensorLayout,
      rgb.with_preprocessing(Preprocessing::new(
        ChannelOrder::Rgb,
        TensorLayout::Nhwc,
        1.0 / 127.5,
        [-1.0, -1.0, -1.0],
      )),
    ),
    (
      EmbeddingSpaceField::PreprocessingScale,
      rgb.with_preprocessing(Preprocessing::new(
        ChannelOrder::Rgb,
        TensorLayout::Nchw,
        1.0 / 255.0,
        [-1.0, -1.0, -1.0],
      )),
    ),
    (
      EmbeddingSpaceField::PreprocessingBias,
      rgb.with_preprocessing(Preprocessing::new(
        ChannelOrder::Rgb,
        TensorLayout::Nchw,
        1.0 / 127.5,
        [0.0, -1.0, -1.0],
      )),
    ),
  ] {
    let far = normalise_row(&[1.0, 0.0], 0, space_of(1, &other)).expect("unit");
    let error = a
      .cosine(&far)
      .expect_err("a different manifest is a different space");
    assert!(
      matches!(error, Error::IncomparableEmbeddings(p) if p.field() == want),
      "expected IncomparableEmbeddings({want}), got {error:?}"
    );
  }
}

#[test]
fn a_non_finite_preprocessing_scale_still_names_one_space() {
  // The space check compares the manifest's `f32`s by a CANONICAL bit pattern.
  // Under `==` a NaN scale would fail to equal itself, and an embedding would
  // be refused against its own twin for a reason having nothing to do with
  // either embedding.
  //
  // This is about `Preprocessing`'s own `Eq` LAWFULNESS and nothing more: the
  // type is public with `const` constructors, so a NaN one can be built, and a
  // `PartialEq` that is not reflexive is not an equivalence relation. It is
  // not about a broken manifest surviving to a comparison — `load` refuses one
  // (`a_manifest_whose_preprocessing_is_not_finite_is_refused_at_load`), so no
  // space this crate stamps can reach here with a NaN in it. The spaces below
  // are therefore built through `EmbeddingSpace::of` directly.
  let with_nan = |payload: f32| {
    model(2).with_preprocessing(Preprocessing::new(
      ChannelOrder::Rgb,
      TensorLayout::Nchw,
      payload,
      [payload, -1.0, -1.0],
    ))
  };
  let nan = with_nan(f32::NAN);
  let a = normalise_row(&[1.0, 0.0], 0, space_of(1, &nan)).expect("unit");
  // Deliberately a DIFFERENT NaN payload, not a second copy of the same
  // constant: a raw `to_bits` comparison passes the same-constant case and
  // fails this one, so the same-constant case alone cannot tell a canonical
  // relation from a bitwise one. `f32` has 2²³ − 1 quiet NaN payloads and an
  // arithmetic result may be any of them.
  let other_payload = f32::from_bits(f32::NAN.to_bits() | 1);
  assert!(other_payload.is_nan() && other_payload.to_bits() != f32::NAN.to_bits());
  let b = normalise_row(&[1.0, 0.0], 0, space_of(1, &with_nan(other_payload))).expect("unit");
  assert_eq!(
    a.cosine(&b)
      .expect("one manifest is one space, whatever it holds"),
    1.0
  );
  assert!(
    a.cosine(&normalise_row(&[1.0, 0.0], 0, space(2)).expect("unit"))
      .is_err(),
    "a NaN scale is still a DIFFERENT space from a finite one"
  );
}

#[test]
fn an_embedding_carries_the_space_that_produced_it() {
  // Deliberately NOT the default preprocessing. A manifest built from
  // `input`/`output`/`dim` alone reconstructs `Preprocessing::ARCFACE`, so a
  // gate written against the default one would pass while the preprocessing
  // half — the half that decides the space and that no shape check can see —
  // was silently replaced.
  let manifest = model(2).with_preprocessing(Preprocessing::from_mean_and_divisor(
    ChannelOrder::Bgr,
    TensorLayout::Nhwc,
    [104.0, 117.0, 123.0],
    58.0,
  ));
  assert_ne!(manifest.preprocessing(), Preprocessing::ARCFACE);
  let embedding = normalise_row(&[1.0, 0.0], 0, space_of(1, &manifest)).expect("unit");
  assert_eq!(embedding.space(), space_of(1, &manifest));
  assert_eq!(
    embedding.space().preprocessing(),
    manifest.preprocessing(),
    "the preprocessing half of the space must travel with the vector too"
  );
  assert_eq!(embedding.space().dim(), embedding.dim());
}

#[test]
fn a_manifest_carries_its_own_preprocessing() {
  let manifest = FaceModel::new("data", "embedding", 512);
  assert_eq!(manifest.input(), "data");
  assert_eq!(manifest.output(), "embedding");
  assert_eq!(manifest.dim(), 512);
  assert_eq!(manifest.preprocessing(), Preprocessing::ARCFACE);

  let adaface = manifest.with_preprocessing(Preprocessing::from_mean_and_divisor(
    ChannelOrder::Bgr,
    TensorLayout::Nchw,
    [127.5, 127.5, 127.5],
    127.5,
  ));
  assert_eq!(adaface.preprocessing().order(), ChannelOrder::Bgr);
  assert_eq!(
    manifest.preprocessing().order(),
    ChannelOrder::Rgb,
    "the builder must not mutate the manifest it was called on"
  );
}

#[test]
fn options_default_to_the_module_default() {
  assert_eq!(FaceEmbedderOptions::new().compute(), DEFAULT_FACE_COMPUTE);
  assert_eq!(FaceEmbedderOptions::default(), FaceEmbedderOptions::new());
  assert_eq!(
    FaceEmbedderOptions::new()
      .with_compute(crate::ComputeUnits::CpuOnly)
      .compute(),
    crate::ComputeUnits::CpuOnly
  );
}

#[cfg(feature = "serde")]
#[test]
fn options_round_trip_through_serde() {
  let options = FaceEmbedderOptions::new().with_compute(crate::ComputeUnits::CpuAndGpu);
  let json = serde_json::to_string(&options).expect("serialisable");
  let back: FaceEmbedderOptions = serde_json::from_str(&json).expect("deserialisable");
  assert_eq!(back, options);
  let defaulted: FaceEmbedderOptions = serde_json::from_str("{}").expect("compute defaults");
  assert_eq!(defaulted.compute(), DEFAULT_FACE_COMPUTE);
}

#[cfg(feature = "serde")]
#[test]
fn preprocessing_round_trips_through_serde() {
  // The manifest's feature names are a compile-time contract, but the
  // preprocessing is the part that differs between artifacts and is the part
  // a deployment may want to carry in configuration.
  let preprocessing =
    Preprocessing::new(ChannelOrder::Bgr, TensorLayout::Nhwc, 0.5, [1.0, 2.0, 3.0]);
  let json = serde_json::to_string(&preprocessing).expect("serialisable");
  assert!(
    json.contains("bgr"),
    "the channel order should be kebab-case: {json}"
  );
  let back: Preprocessing = serde_json::from_str(&json).expect("deserialisable");
  assert_eq!(back, preprocessing);
}

#[test]
fn the_tensor_built_has_the_rank_the_model_declared() {
  // `resolve_batch` accepts the UNBATCHED rank-3 form that real ArcFace
  // exports declare, and used to keep only the numeric capacity from it — so a
  // model that loads as supported was then always handed a leading-batch
  // rank-4 tensor, and every single prediction failed. A contract the loader
  // accepted has to be one `build_input` can actually satisfy.
  for (declared, layout) in [
    (
      vec![3usize, TEMPLATE_SIZE, TEMPLATE_SIZE],
      TensorLayout::Nchw,
    ),
    (
      vec![TEMPLATE_SIZE, TEMPLATE_SIZE, 3usize],
      TensorLayout::Nhwc,
    ),
    (
      vec![8usize, 3, TEMPLATE_SIZE, TEMPLATE_SIZE],
      TensorLayout::Nchw,
    ),
    (
      vec![4usize, TEMPLATE_SIZE, TEMPLATE_SIZE, 3],
      TensorLayout::Nhwc,
    ),
  ] {
    let batch = if declared.len() == 4 { declared[0] } else { 1 };
    let (contract, _) = check(&graph(&declared, &[batch, DIM]), &manifest(layout))
      .expect("a shape real exports declare must load");
    assert_eq!(
      input_shape(contract, layout),
      declared,
      "the loader accepted {declared:?} and would then feed the graph a different shape"
    );
  }
}

#[test]
fn a_transposed_output_tensor_is_refused() {
  // The silent one. `[dim, batch]` has exactly the same element COUNT as
  // `[batch, dim]`, so a count-only check passes it, and the flattening that
  // follows then cuts `dim`-sized chunks across the WRONG axis — mixing
  // components between faces and returning embeddings that are plausible,
  // unit-norm and wrong. No shape check, no finiteness check and no cosine can
  // see it afterwards.
  let error = check_predicted_shape(&[512, 4], 4 * 512, OutputContract::Batched, 4, 512, 4 * 512)
    .expect_err("a [dim, batch] tensor is not a [batch, dim] tensor");
  assert!(
    matches!(&error, Error::OutputShape(payload) if payload.got() == [512, 4]),
    "expected OutputShape([512, 4]), got {error:?}"
  );

  // The contract's own shape still passes, and so does the batch-one `[dim]`
  // form — but ONLY for a batch-one contract.
  assert!(
    check_predicted_shape(&[4, 512], 4 * 512, OutputContract::Batched, 4, 512, 4 * 512).is_ok()
  );
  assert!(check_predicted_shape(&[512], 512, OutputContract::Flat, 1, 512, 512).is_ok());

  // And a declared form is binding: a graph that promised [batch, dim] does
  // not get to emit [dim] instead. There is no third arm softening that any
  // more — the `Undeclared` form these lines used to cover is refused at load,
  // so a predicted tensor is always measured against a shape the graph named.
  assert!(check_predicted_shape(&[512], 512, OutputContract::Batched, 1, 512, 512).is_err());
}

#[test]
fn a_diverging_element_count_is_not_reported_as_a_shape_mismatch() {
  // FALSIFIER (red first, on CONTENTS). `count` is CoreML's OWN answer rather
  // than a product of the cached shape — which is the entire reason it is
  // checked alongside the axes. So the two can disagree, and when only the
  // COUNT does, the shape matched: the old payload put the same vector in
  // both fields and rendered "expected [4, 512], got [4, 512]", a shape
  // mismatch that did not happen.
  let error = check_predicted_shape(
    &[4, 512],
    4 * 512 - 1,
    OutputContract::Batched,
    4,
    512,
    4 * 512,
  )
  .expect_err("an element count short of the contract is a divergence");
  let message = error.to_string();
  assert!(
    !message.contains("expected [4, 512], got [4, 512]"),
    "the shapes are equal; reporting them as a mismatch is a falsehood, got {message:?}"
  );
  assert!(
    message.contains("2047") && message.contains("2048"),
    "the failure must name the counts that diverged, got {message:?}"
  );
  assert!(
    matches!(
      &error,
      Error::OutputElementCount(payload) if payload.got() == 2047 && payload.expected() == 2048
    ),
    "the payload must carry both counts, got {error:?}"
  );

  // A genuine axis divergence is still an `OutputShape`, so splitting the two
  // did not swallow the one that matters.
  assert!(
    matches!(
      check_predicted_shape(&[512, 4], 4 * 512, OutputContract::Batched, 4, 512, 4 * 512),
      Err(Error::OutputShape(_))
    ),
    "a transposed tensor is still a shape mismatch"
  );
}

#[test]
fn normalising_survives_components_an_f32_square_cannot_hold() {
  // The squared norm used to accumulate in `f32`, where `v * v` overflows to
  // infinity for a large component and underflows to zero for a small one.
  // Both reported `EmbeddingZero` — "this row has no direction" — for a row
  // with a perfectly good direction. The magnitudes are the artifact's, not
  // ours: nothing in the contract says a model's pre-normalisation output is
  // near unit scale.
  // Both components are finite `f32`s, and so is their norm (3.0e38); only the
  // SQUARES leave the type.
  let big =
    normalise_row(&[1.8e38, 2.4e38], 0, space(2)).expect("a large but finite row has a direction");
  assert!(
    (big.as_slice()[0] - 0.6).abs() < 1e-6,
    "got {:?}",
    big.as_slice()
  );
  assert!(
    (big.as_slice()[1] - 0.8).abs() < 1e-6,
    "got {:?}",
    big.as_slice()
  );

  let small =
    normalise_row(&[3.0e-25, 4.0e-25], 1, space(2)).expect("a tiny but finite row has a direction");
  assert!(
    (small.as_slice()[0] - 0.6).abs() < 1e-6,
    "got {:?}",
    small.as_slice()
  );
  assert!(
    (small.as_slice()[1] - 0.8).abs() < 1e-6,
    "got {:?}",
    small.as_slice()
  );

  // Only an EXACT zero magnitude is a genuine absence of direction.
  assert!(matches!(
    normalise_row(&[0.0, -0.0], 2, space(2)).expect_err("zero has no direction"),
    Error::EmbeddingZero(_)
  ));
}

#[test]
fn the_load_time_contract_requires_an_f32_multi_array() {
  // Names and shapes were checked; the tensor KIND and element type never
  // were, while inference always supplies and extracts `f32` multi-arrays. An
  // f16 export therefore loaded clean and failed every prediction. Both
  // clauses belong to the contract now, on both features, so neither is a call
  // this door could stop making.
  let nchw = manifest(TensorLayout::Nchw);
  let shape = [4, 3, TEMPLATE_SIZE, TEMPLATE_SIZE];
  for wrong in [DataType::F16, DataType::F64, DataType::I32] {
    let description = ModelDescription::from_parts(
      vec![fixed("data", &shape, wrong)],
      vec![fixed("embedding", &[4, DIM], DataType::F32)],
      Vec::new(),
    );
    let error = check(&description, &nchw).unwrap_err();
    assert!(
      matches!(&error, Error::ContractMismatch(m) if m.feature() == "data"),
      "a {wrong} input must not load against an f32 inference path: {error}"
    );

    let description = ModelDescription::from_parts(
      vec![fixed("data", &shape, DataType::F32)],
      vec![fixed("embedding", &[4, DIM], wrong)],
      Vec::new(),
    );
    let error = check(&description, &nchw).unwrap_err();
    assert!(
      matches!(&error, Error::ContractMismatch(m) if m.feature() == "embedding"),
      "a {wrong} output must not load against an f32 inference path: {error}"
    );
  }

  // `data_type()` is `None` exactly when the feature is NOT a multi-array —
  // the case the module doc's own census hits, since both third-party CoreML
  // ArcFace builds it surveys declare `ImageType` inputs. Such a feature
  // reports no shape either, so it is refused one clause EARLIER than its
  // dtype, by the same rank clause that refuses an undeclared shape.
  let image = FeatureInfo::from_parts("data".to_string(), Vec::new(), None, false, None);
  assert_eq!(image.data_type(), None);
  assert_eq!(image.shape_constraint(), None);
  let description = ModelDescription::from_parts(
    vec![image],
    vec![fixed("embedding", &[4, DIM], DataType::F32)],
    Vec::new(),
  );
  assert!(
    matches!(check(&description, &nchw), Err(Error::ContractMismatch(m)) if m.feature() == "data"),
    "a feature that is not a multi-array must not resolve to batch 1"
  );

  // The all-f32 description still loads.
  assert!(check(&graph(&shape, &[4, DIM]), &nchw).is_ok());
}

// ── The element counts the artifact's batch decides ────────────────────

/// **FALSIFIER (red first).** The batch is the ARTIFACT's — `Dim::AnyFixed`
/// reads back whatever the graph pins — and this description used to load
/// clean: every clause of the contract passes, because a `usize::MAX / 1000`
/// axis is a perfectly well-formed pinned dimension.
///
/// What followed was not an error. `build_input` computed
/// `batch · 3 · 112 · 112`, which for this batch is `6.9e20` against a `usize`
/// ceiling of `1.8e19`; in a release build the product wraps to
/// `11_658_342_254_584_413_440`, `vec![0.0f32; …]` then aborts on that, and for
/// a batch chosen to wrap SMALL the allocation succeeds and the first
/// `row * FACE_ELEMENTS ..` slice panics out of the too-short buffer. Either
/// way a model this door ACCEPTED terminates the caller.
///
/// The assertion is the REFUSAL rather than the arithmetic, which is what makes
/// it independent of `-C overflow-checks`: replace `checked_mul` with `*` and a
/// debug build panics inside the multiply while a release build returns `Ok` —
/// both red here, for the same reason, without the gate knowing which profile
/// it is in.
#[test]
fn a_batch_whose_tensor_element_count_leaves_usize_is_refused_at_load() {
  // The INPUT's count is the artifact's alone: `batch · 112 · 112 · 3`.
  let batch = usize::MAX / 1000;
  assert!(
    batch.checked_mul(DIM).is_some(),
    "this batch must overflow the INPUT count only, or the case below is the \
     one being tested twice"
  );
  let description = graph(&[batch, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], &[batch, DIM]);
  let error = check(&description, &model(DIM))
    .expect_err("a batch whose input tensor cannot be counted must not load");
  assert!(
    matches!(
      &error,
      Error::ElementCountOverflow(o)
        if o.tensor() == PredictionTensor::Input
          && o.batch() == batch
          && o.per_row() == TEMPLATE_BYTES
    ),
    "{error}"
  );

  // The OUTPUT's count pairs that batch with the MANIFEST's width, so it can
  // overflow where the input's does not — a second clause, not the same one.
  let dim = 1 << 40;
  let batch = usize::MAX / dim + 1;
  assert!(
    batch.checked_mul(TEMPLATE_BYTES).is_some(),
    "this batch must overflow the OUTPUT count only"
  );
  let description = graph(&[batch, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], &[batch, dim]);
  let error = check(&description, &model(dim))
    .expect_err("a batch and width whose output tensor cannot be counted must not load");
  assert!(
    matches!(
      &error,
      Error::ElementCountOverflow(o)
        if o.tensor() == PredictionTensor::Output && o.batch() == batch && o.per_row() == dim
    ),
    "{error}"
  );

  // And a batch that counts fine still loads, so the refusal is the overflow
  // and not the size.
  let (input, _) = check(
    &graph(&[8, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], &[8, DIM]),
    &model(DIM),
  )
  .expect("a batch-8 export counts to 301056 input and 4096 output elements");
  assert_eq!(input.batch, 8);
}

/// **FALSIFIER (red first).** Fitting `usize` is strictly weaker than the
/// memory existing, so the count proved at load does not finish the job: a
/// batch of `2⁵⁵` counts fine and asks for petabytes.
///
/// The batch that reaches that regime cannot be allocated in a test — that is
/// the point of it — so the helper is driven directly at two absurd lengths,
/// one on each side of `Vec`'s own capacity arithmetic. `vec![0.0f32; n]`
/// answers both by ABORTING the process, which no `expect_err` can observe and
/// no caller can handle; the assertion here is that an `Err` comes back at all.
#[test]
fn the_tensor_allocator_refuses_rather_than_aborting() {
  // `usize::MAX` f32s is `usize::MAX · 4` bytes: the length cannot even be
  // turned into a layout, and `Vec` reports `CapacityOverflow`.
  let error = zeroed_tensor(PredictionTensor::Output, usize::MAX)
    .expect_err("a length whose byte size leaves `usize` has no buffer");
  assert!(
    matches!(
      &error,
      Error::AllocationFailed(a)
        if a.tensor() == PredictionTensor::Output && a.elements() == usize::MAX
    ),
    "{error}"
  );

  // `usize::MAX / 8` f32s is a layout `Vec` will happily describe — under
  // `isize::MAX` bytes — and that no allocator will satisfy, so this is the
  // arm that reaches a real `AllocError`.
  let beyond_memory = usize::MAX / 8;
  let error = zeroed_tensor(PredictionTensor::Input, beyond_memory)
    .expect_err("a buffer the allocator refuses is an error, not an abort");
  assert!(
    matches!(
      &error,
      Error::AllocationFailed(a)
        if a.tensor() == PredictionTensor::Input && a.elements() == beyond_memory
    ),
    "{error}"
  );
  assert!(
    error.to_string().contains(&beyond_memory.to_string()),
    "the refusal must name the length that was asked for, got {error}"
  );

  // A length that CAN be met still comes back zeroed and exactly that long,
  // so the fallible path did not change what the buffer is.
  let data = zeroed_tensor(PredictionTensor::Input, FACE_ELEMENTS).expect("one face fits");
  assert_eq!(data.len(), FACE_ELEMENTS);
  assert!(data.iter().all(|v| v.to_bits() == 0.0f32.to_bits()));
}

/// **FALSIFIER (red first).** The sibling of the gate above, and the reason
/// round 7 was not the whole class: the two per-PREDICTION buffers were made
/// fallible and the per-ROW one was not.
///
/// `normalise_row` collected its components into a `Box<[f32]>` — a `collect`
/// over a `TrustedLen` iterator, so `Vec::with_capacity(row.len())` under the
/// covers, so `handle_alloc_error` and an ABORT when the allocator refuses.
/// The width is the MANIFEST's `dim`, the same number `elements.output` is
/// half of, and this allocation is per row: across one chunk it duplicates the
/// whole output tensor while the flat gather buffer and both native tensors
/// are still live. A large but valid artifact therefore terminated the caller
/// AFTER the fallibly allocated flat buffer had succeeded.
///
/// The row is a slice, so an absurd width cannot be driven through
/// `normalise_row` itself — the slice would have to exist. The helper it
/// reserves through is driven directly instead, exactly as
/// `the_tensor_allocator_refuses_rather_than_aborting` drives `zeroed_tensor`,
/// at two lengths on either side of `Vec`'s capacity arithmetic.
#[test]
fn the_embedding_row_allocator_refuses_rather_than_aborting() {
  // `usize::MAX` f32s is `usize::MAX * 4` bytes: the length cannot even be
  // turned into a layout, and `Vec` reports `CapacityOverflow`.
  let error =
    embedding_buffer(usize::MAX).expect_err("a width whose byte size leaves `usize` has no row");
  assert!(
    matches!(
      &error,
      Error::AllocationFailed(a)
        if a.tensor() == PredictionTensor::Output && a.elements() == usize::MAX
    ),
    "{error}"
  );

  // `usize::MAX / 8` f32s is a layout `Vec` will happily describe — under
  // `isize::MAX` bytes — and that no allocator will satisfy, so this is the
  // arm that reaches a real `AllocError`.
  let beyond_memory = usize::MAX / 8;
  let error = embedding_buffer(beyond_memory)
    .expect_err("a row the allocator refuses is an error, not an abort");
  assert!(
    matches!(
      &error,
      Error::AllocationFailed(a)
        if a.tensor() == PredictionTensor::Output && a.elements() == beyond_memory
    ),
    "{error}"
  );
  assert!(
    error.to_string().contains(&beyond_memory.to_string()),
    "the refusal must name the width that was asked for, got {error}"
  );

  // A width that CAN be met still normalises to the same vector, so the
  // fallible reservation did not change what an embedding is.
  let embedding = normalise_row(&[3.0, 4.0], 0, space(2)).expect("finite and nonzero");
  assert_eq!(embedding.dim(), 2);
  assert!((embedding.as_slice()[0] - 0.6).abs() < 1e-6);
  assert!((embedding.as_slice()[1] - 0.8).abs() < 1e-6);
}

// ── The one gate here that loads a real artifact ───────────────────────────

/// **The wiring, on a description CoreML itself produced.**
///
/// Every other gate in this file drives [`load_contract`] over a fixture. This
/// one runs [`FaceEmbedder::load`] end to end against
/// `Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc`, which is COMMITTED
/// — 1.1 MiB, staged by no download — so unlike everything else in this
/// repository that loads a model it carries no `#[ignore]`. Silero is a real,
/// fixed-shape, six-feature graph that is simply not this door's model, which
/// is the exact shape of a mis-pointed `model_path`.
///
/// What it pins that the fixtures cannot: that the decision runs where `load`
/// puts it, over a snapshot the CoreML runtime built rather than one
/// `from_parts` assembled; and that a refused load never reaches
/// `digest_artifact`, since the error is a contract mismatch and not a digest
/// failure.
///
/// What it CANNOT pin, stated rather than implied: silero declares no rank-3 or
/// rank-4 input, so both refusals below land in `load_contract` before
/// [`Checked::new`] is reached, and no committed artifact is shaped like a face
/// model. The check inside `Checked::new` therefore has no real-model gate on
/// THIS door — what makes it undeletable here is the `Checked` field, which is
/// a compile-time fact rather than a test. (`audio::identity`'s own silero gate
/// does reach `Checked::new`, over the same bundle.)
#[test]
fn the_face_door_refuses_the_vendored_silero_bundle() {
  let bundle = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
    .join("../Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc");
  assert!(
    bundle.is_dir(),
    "the vendored silero bundle is committed, so this gate is NOT model-gated; looked for {}",
    bundle.display()
  );
  let options = FaceEmbedderOptions::new().with_compute(ComputeUnits::CpuOnly);

  // A manifest naming features silero does not declare: the by-name clause,
  // with the names it DOES declare in the message.
  let error = FaceEmbedder::load(&bundle, model(DIM), options)
    .expect_err("silero declares no `data` feature");
  assert!(
    matches!(&error, Error::ContractMismatch(m)
      if m.feature() == "data" && m.actual().contains("audio_input")),
    "{error}"
  );

  // A manifest naming features silero DOES declare, so the lookup succeeds and
  // the geometry is what refuses it: `audio_input` is `[1, 4160]`, a rank no
  // contract of this door's can be built from.
  let error = FaceEmbedder::load(
    &bundle,
    FaceModel::new("audio_input", "vad_output", DIM),
    options,
  )
  .expect_err("silero's audio window is not a template face");
  assert!(
    matches!(&error, Error::ContractMismatch(m)
      if m.feature() == "audio_input" && m.actual() == "[1, 4160]"),
    "{error}"
  );
}

/// **Where the walk sits in `load`, pinned by what a refusal is CALLED.**
///
/// The digest is taken LAST — after `Model::load` and after the contract has
/// been checked — so a manifest the artifact refuses pays no walk at all, and
/// the value stamped is of the same path CoreML was handed. The order is not
/// directly observable from outside the door: seeing the stamped digest needs a
/// load that SUCCEEDS, and this crate stages no face artifact. What IS
/// observable is the CLASS of a refusal, and it separates the two orders — a
/// walk taken FIRST refuses a path CoreML never saw as a digest failure, where
/// a walk taken last leaves it as CoreML's own `NotFound`.
///
/// The other half of the ordering is on
/// `the_face_door_refuses_the_vendored_silero_bundle` above, whose refusals are
/// `ContractMismatch` and not `ArtifactDigest`: a real 1.1 MiB bundle the
/// manifest does not fit is never hashed.
#[test]
fn a_load_that_cannot_open_the_artifact_never_walks_it() {
  let temp = tempfile::tempdir().expect("tempdir");
  let absent = temp.path().join("not-there.mlmodelc");
  let options = FaceEmbedderOptions::new().with_compute(ComputeUnits::CpuOnly);

  let error =
    FaceEmbedder::load(&absent, model(DIM), options).expect_err("there is nothing to load");
  assert!(
    matches!(&error, Error::Load(crate::LoadError::NotFound(path)) if path == &absent),
    "a path CoreML cannot open must fail as a LOAD, not as a digest of bytes the door never \
     needed; got {error:?}"
  );
}

#[test]
fn manifest_equality_and_space_identity_are_one_relation() {
  // `FaceModel` used to derive `PartialEq` — `f32`'s `==`, under which `+0.0`
  // and `−0.0` are equal — while `space_difference` compared the same `f32`s
  // by RAW BIT PATTERN, under which they are not. One type, two equality
  // relations, contradicting each other; and it was the space check that got
  // it wrong, refusing a pair whose preprocessing is the same function.
  let plus = model(2).with_preprocessing(Preprocessing::new(
    ChannelOrder::Rgb,
    TensorLayout::Nchw,
    1.0 / 127.5,
    [0.0, -1.0, -1.0],
  ));
  let minus = model(2).with_preprocessing(Preprocessing::new(
    ChannelOrder::Rgb,
    TensorLayout::Nchw,
    1.0 / 127.5,
    [-0.0, -1.0, -1.0],
  ));
  assert_ne!(
    plus.preprocessing().bias().map(f32::to_bits),
    minus.preprocessing().bias().map(f32::to_bits),
    "the two must differ in their BITS, or the gate proves nothing"
  );

  let a = normalise_row(&[1.0, 0.0], 0, space_of(1, &plus)).expect("unit");
  let b = normalise_row(&[1.0, 0.0], 0, space_of(1, &minus)).expect("unit");
  for (name, related) in [
    ("FaceModel: PartialEq", plus == minus),
    (
      "Preprocessing: PartialEq",
      plus.preprocessing() == minus.preprocessing(),
    ),
    (
      "EmbeddingSpace: PartialEq",
      space_of(1, &plus) == space_of(1, &minus),
    ),
    ("FaceEmbedding::dot", a.dot(&b).is_ok()),
  ] {
    assert!(
      related,
      "`{name}` says these two are different, but they preprocess identically — every relation \
       on this manifest has to be the same relation"
    );
  }

  // The direction that matters: `write_row` gives byte-identical tensors for
  // these two, so refusing the pair refuses provably equivalent work.
  let mut left = vec![0.0f32; 3 * TEMPLATE_SIZE * TEMPLATE_SIZE];
  let mut right = left.clone();
  write_row(&mut left, &ramp_face(), plus.preprocessing());
  write_row(&mut right, &ramp_face(), minus.preprocessing());
  assert_eq!(
    left.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
    right.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
    "the two manifests preprocess to the same bits"
  );

  // And the relation still SEPARATES what it must: canonicalising `±0.0` and
  // NaN must not collapse two genuinely different biases.
  let other = model(2).with_preprocessing(Preprocessing::new(
    ChannelOrder::Rgb,
    TensorLayout::Nchw,
    1.0 / 127.5,
    [1e-45, -1.0, -1.0],
  ));
  assert_ne!(
    space_of(1, &plus),
    space_of(1, &other),
    "a subnormal bias is not zero"
  );
  assert!(
    a.dot(&normalise_row(&[1.0, 0.0], 0, space_of(1, &other)).expect("unit"))
      .is_err()
  );
}

#[test]
fn a_space_hashes_by_the_relation_it_compares_by() {
  // `Eq` and `Hash` have to agree or an `EmbeddingSpace` silently misbehaves as
  // a map key — and both of this relation's foldings (`±0.0`, and every NaN
  // being one value) are exactly the places a derived `Hash` would disagree
  // with it.
  let hash_of = |model: FaceModel| {
    use core::hash::{Hash, Hasher};
    let mut hasher = std::hash::DefaultHasher::new();
    space_of(1, &model).hash(&mut hasher);
    hasher.finish()
  };
  let with = |bias: [f32; 3], scale: f32| {
    model(2).with_preprocessing(Preprocessing::new(
      ChannelOrder::Rgb,
      TensorLayout::Nchw,
      scale,
      bias,
    ))
  };
  for (left, right) in [
    (with([0.0, -1.0, -1.0], 1.0), with([-0.0, -1.0, -1.0], 1.0)),
    (
      with([-1.0; 3], f32::NAN),
      with([-1.0; 3], f32::from_bits(f32::NAN.to_bits() | 1)),
    ),
  ] {
    assert_eq!(
      space_of(1, &left),
      space_of(1, &right),
      "these name one space"
    );
    assert_eq!(
      hash_of(left),
      hash_of(right),
      "equal spaces must hash equal, or a map keyed by one is broken"
    );
  }

  // The other direction is NOT required by the `Hash` contract — unequal
  // values may collide — but it is the whole reason to key a map by a space,
  // and the artifact is the field most likely to be the only difference. So
  // the three fields that joined the relation are asserted to reach the
  // hasher too.
  let one = model(2);
  let renamed = FaceModel::new("input_1", "var_9", 2);
  for (name, left, right) in [
    ("artifact", space_of(1, &one), space_of(2, &one)),
    ("feature names", space_of(1, &one), space_of(1, &renamed)),
  ] {
    assert_ne!(left, right, "{name}: these are two spaces");
    let digest = |space: EmbeddingSpace| {
      use core::hash::{Hash, Hasher};
      let mut hasher = std::hash::DefaultHasher::new();
      space.hash(&mut hasher);
      hasher.finish()
    };
    assert_ne!(
      digest(left),
      digest(right),
      "{name}: two spaces that differ only here must reach the hasher, or a map keyed by a space \
       buckets unrelated artifacts together"
    );
  }
}

#[test]
fn two_heads_of_one_artifact_are_two_spaces() {
  // The case that makes an output feature name NOT routing. A graph with two
  // `[batch, dim]` heads — an embedding and, say, a projection or an
  // auxiliary logit block of the same width — is one artifact, one input, one
  // preprocessing and one width. The output name is the only thing that says
  // WHICH FUNCTION produced the numbers, and the two functions are unrelated.
  //
  // Scored, this pair reads as a face comparison. Nothing else in the space
  // can see the difference: the digest is equal because the bytes ARE equal.
  let one_artifact = artifact(1);
  let embedding = FaceModel::new("data", "embedding", 2);
  let projection = FaceModel::new("data", "projection", 2);
  assert_eq!(
    (embedding.input(), embedding.dim()),
    (projection.input(), projection.dim()),
    "the two heads must differ ONLY in the output name, or this gate proves nothing"
  );

  let head =
    normalise_row(&[1.0, 0.0], 0, EmbeddingSpace::of(one_artifact, &embedding)).expect("unit");
  let other_head = normalise_row(
    &[1.0, 0.0],
    0,
    EmbeddingSpace::of(one_artifact, &projection),
  )
  .expect("unit");

  let error = head
    .dot(&other_head)
    .expect_err("two heads of one graph are two spaces");
  assert!(
    matches!(
      error,
      Error::IncomparableEmbeddings(p) if p.field() == EmbeddingSpaceField::OutputFeature
    ),
    "expected IncomparableEmbeddings(OutputFeature), got {error:?}"
  );
}

#[test]
fn two_artifacts_with_one_schema_are_two_spaces() {
  // The residual a previous round STATED and left open: "two distinct
  // artifacts with one schema are one space as far as this type can see".
  // They are not one space — the trained parameters are most of the function
  // that produced the vector — and the digest of the bytes `load` read is what
  // closes it.
  //
  // A fine-tune, a requantisation, or an unrelated 512-wide ArcFace-family
  // export all land here: same width, same feature names, same preprocessing,
  // different weights. Scored, they read `1.0` — a perfect match between two
  // vectors that mean nothing to each other.
  let schema = model(2);
  let first = normalise_row(&[1.0, 0.0], 0, space_of(1, &schema)).expect("unit");
  let second = normalise_row(&[1.0, 0.0], 0, space_of(2, &schema)).expect("unit");
  assert_ne!(
    first.space().artifact(),
    second.space().artifact(),
    "the two artifacts must differ in their digest, or this gate proves nothing"
  );
  assert_eq!(
    (first.space().dim(), first.space().preprocessing()),
    (second.space().dim(), second.space().preprocessing()),
    "and they must agree on everything else, or something weaker than the digest could refuse"
  );

  let error = first
    .dot(&second)
    .expect_err("two artifacts are two spaces whatever their schemas say");
  assert!(
    matches!(
      error,
      Error::IncomparableEmbeddings(p) if p.field() == EmbeddingSpaceField::Artifact
    ),
    "expected IncomparableEmbeddings(Artifact), got {error:?}"
  );

  // And the digest is not a per-LOAD token: two embedders over the same bytes
  // are one space, which is what `&self` fan-out needs.
  let same_bytes = normalise_row(&[1.0, 0.0], 0, space_of(1, &schema)).expect("unit");
  assert_eq!(
    first
      .dot(&same_bytes)
      .expect("the same artifact read twice is one space"),
    1.0
  );
}

#[test]
fn feature_names_select_which_tensor_and_therefore_do_decide_the_space() {
  // The INVERSE of what this gate asserted a round ago, and the reversal is
  // the useful part. The old argument was that a feature name is the string
  // CoreML routes a tensor by, so renaming a graph's features re-exports the
  // same weights and the vectors stay in one space.
  //
  // That is right about renaming and wrong about names. For a model with two
  // `[batch, dim]` heads the output name selects WHICH FUNCTION produced the
  // numbers — see `two_heads_of_one_artifact_are_two_spaces` — and no other
  // field can tell those two apart. What the old argument actually wanted was
  // a way to see that two exports hold the same weights, and that is the
  // artifact digest's job, not the name's.
  let renamed = FaceModel::new("input_1", "var_2011", 2);
  let original = model(2);
  let a = normalise_row(&[1.0, 0.0], 0, space_of(1, &original)).expect("unit");
  let b = normalise_row(&[1.0, 0.0], 0, space_of(1, &renamed)).expect("unit");

  let error = a
    .dot(&b)
    .expect_err("differently routed tensors are differently produced numbers");
  assert!(
    matches!(
      error,
      Error::IncomparableEmbeddings(p) if p.field() == EmbeddingSpaceField::InputFeature
    ),
    "expected IncomparableEmbeddings(InputFeature) — the first field that differs — got {error:?}"
  );

  // The cost of the reversal, stated as an assertion rather than left in a
  // paragraph: a NUMERICALLY IDENTICAL re-export under other names is now
  // refused. Under this crate's provenance model that is loud and correct —
  // `MODELS_LOCK` already treats bundle bytes as identity, and two files that
  // are not the same bytes are not the same artifact — but it is a refusal
  // where a round ago there was a score, and a caller who re-exports has to
  // re-embed.
  let re_exported = FaceModel::new("input_1", "var_2011", 2);
  let same_weights_new_names =
    normalise_row(&[1.0, 0.0], 0, space_of(1, &re_exported)).expect("unit");
  assert!(
    a.dot(&same_weights_new_names).is_err(),
    "a re-export under other names is refused, not scored"
  );
}

#[test]
fn the_space_is_half_the_callers_and_half_the_artifacts() {
  // Also an inversion. This gate used to assert that every field of every
  // stamped space is a value the caller chose — which was true when the space
  // was a projection of the manifest, and was written down precisely because
  // an earlier round had claimed the opposite and rested the whole guarantee
  // on it.
  //
  // Half of it is still true, and the half that is not is the point of the
  // digest: a caller chooses WHICH artifact to load, not what its bytes hash
  // to. `ArtifactDigest` has no public constructor and `EmbeddingSpace` has no
  // public constructor, so the artifact half of a space is a fact about bytes
  // this crate read.
  let caller_built = FaceModel::new("data", "embedding", 2).with_preprocessing(Preprocessing::new(
    ChannelOrder::Bgr,
    TensorLayout::Nhwc,
    1.0 / 128.0,
    [-1.0, -1.0, -1.0],
  ));
  let space = space_of(1, &caller_built);
  let embedding = normalise_row(&[1.0, 0.0], 0, space).expect("unit");

  // The caller's half, exactly as they stated it.
  assert_eq!(embedding.space().input(), caller_built.input());
  assert_eq!(embedding.space().output(), caller_built.output());
  assert_eq!(embedding.space().dim(), caller_built.dim());
  assert_eq!(
    embedding.space().preprocessing(),
    caller_built.preprocessing()
  );

  // THE RESIDUAL THIS GATE USED TO ASSERT, NOW CLOSED. Two distinct artifacts
  // declaring the same width and the same preprocessing used to be one space
  // as far as this crate could see, and their cosine was returned rather than
  // refused — "nothing short of holding the weights closes it". Hashing the
  // weights is holding them for this purpose.
  let second_artifact = normalise_row(&[1.0, 0.0], 0, space_of(2, &caller_built)).expect("unit");
  let error = embedding
    .dot(&second_artifact)
    .expect_err("schema equality is not artifact identity, and the space now knows it");
  assert!(
    matches!(
      error,
      Error::IncomparableEmbeddings(p) if p.field() == EmbeddingSpaceField::Artifact
    ),
    "expected IncomparableEmbeddings(Artifact), got {error:?}"
  );

  // What a caller still cannot do, and where the guarantee has always actually
  // lived: assemble the VECTOR. `FaceEmbedding` has no public constructor, so
  // the components came out of a real prediction by an embedder this crate
  // loaded, and the space stamped on them is the one that embedder ran in.
  assert_eq!(embedding.space(), space);
}
