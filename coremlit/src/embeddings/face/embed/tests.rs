//! Unit gates for the manifest-driven preprocessing and the batch plumbing.
//!
//! No model is needed: every function these exercise is pure. That is the
//! point of putting preprocessing in a manifest — the part that silently
//! degrades an embedding is testable without an artifact.

use super::*;
use crate::embeddings::face::align::TEMPLATE_BYTES;

/// Every real face artifact declares an f32 multi-array for both features.
const F32: Option<DataType> = Some(DataType::F32);

/// A manifest of the given width, standing in for the artifact an embedder
/// would have been loaded against. Every `FaceEmbedding` carries one, so a
/// unit gate has to name one too.
fn space(dim: usize) -> FaceModel {
  FaceModel::new("data", "embedding", dim)
}

/// The resolved input contract as a comparable pair, so a gate can assert the
/// declared RANK and not only the numeric capacity.
fn contract_of(shape: &[usize], layout: TensorLayout) -> Option<(usize, InputRank)> {
  resolve_input_contract(shape, F32, layout).map(|c| (c.batch(), c.rank))
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

#[test]
fn the_input_contract_accepts_the_three_shapes_real_exports_declare() {
  // The rank is asserted alongside the capacity: the two rank-3 rows below
  // used to resolve to a bare `1`, indistinguishable from the batched
  // `[1, 3, 112, 112]` form, and that discarded bit is what fed them a tensor
  // their graph cannot accept.
  assert_eq!(
    contract_of(&[], TensorLayout::Nchw),
    Some((1, InputRank::Batched))
  );
  assert_eq!(
    contract_of(&[3, 112, 112], TensorLayout::Nchw),
    Some((1, InputRank::Unbatched))
  );
  assert_eq!(
    contract_of(&[8, 3, 112, 112], TensorLayout::Nchw),
    Some((8, InputRank::Batched))
  );
  assert_eq!(
    contract_of(&[112, 112, 3], TensorLayout::Nhwc),
    Some((1, InputRank::Unbatched))
  );
  assert_eq!(
    contract_of(&[4, 112, 112, 3], TensorLayout::Nhwc),
    Some((4, InputRank::Batched))
  );
}

#[test]
fn the_input_contract_refuses_a_shape_that_is_not_a_template_face() {
  // The layouts must not accept each other's shapes: that swap is exactly the
  // silent-degradation failure the manifest exists to prevent.
  assert_eq!(contract_of(&[1, 112, 112, 3], TensorLayout::Nchw), None);
  assert_eq!(contract_of(&[1, 3, 112, 112], TensorLayout::Nhwc), None);
  assert_eq!(contract_of(&[1, 3, 96, 112], TensorLayout::Nchw), None);
  assert_eq!(contract_of(&[1, 1, 3, 112, 112], TensorLayout::Nchw), None);
  assert_eq!(contract_of(&[112, 112], TensorLayout::Nchw), None);
  // A declared batch of zero would make `embed` chunk by zero.
  assert_eq!(contract_of(&[0, 3, 112, 112], TensorLayout::Nchw), None);
}

#[test]
fn output_shape_check_binds_batch_and_dim() {
  // The resolved FORM is returned, not just an ok/err: it is what the
  // predicted tensor is then measured against on every call.
  assert_eq!(
    check_output_contract(&[], F32, 4, 512),
    Ok(OutputContract::Undeclared)
  );
  assert_eq!(
    check_output_contract(&[512], F32, 1, 512),
    Ok(OutputContract::Flat)
  );
  assert_eq!(
    check_output_contract(&[4, 512], F32, 4, 512),
    Ok(OutputContract::Batched)
  );
  assert!(check_output_contract(&[4, 128], F32, 4, 512).is_err());
  assert!(check_output_contract(&[2, 512], F32, 4, 512).is_err());
  assert!(check_output_contract(&[1, 4, 512], F32, 4, 512).is_err());
  // A bare `[dim]` is a batch-one form. Declared against a batch of 4 it is
  // not a shorthand, it is a contradiction, and accepting it would leave the
  // predicted-tensor check with two incompatible shapes to allow.
  assert!(
    check_output_contract(&[512], F32, 4, 512).is_err(),
    "[512] must not resolve against a batch-4 graph"
  );
}

#[test]
fn normalising_produces_a_unit_vector() {
  let embedding = normalise_row(&[3.0, 4.0], 0, space(2).space()).expect("finite and nonzero");
  assert_eq!(embedding.dim(), 2);
  assert!((embedding.as_slice()[0] - 0.6).abs() < 1e-6);
  assert!((embedding.as_slice()[1] - 0.8).abs() < 1e-6);
  let norm: f32 = embedding.as_slice().iter().map(|v| v * v).sum();
  assert!((norm - 1.0).abs() < 1e-6);
  assert_eq!(embedding.to_vec(), embedding.as_slice().to_vec());
}

#[test]
fn a_zero_row_is_refused_and_names_the_callers_index() {
  let error =
    normalise_row(&[0.0, 0.0, 0.0], 7, space(3).space()).expect_err("zero has no direction");
  assert!(
    matches!(error, Error::EmbeddingZero(payload) if payload.row() == 7),
    "expected EmbeddingZero(7), got {error:?}"
  );
}

#[test]
fn a_non_finite_row_names_the_row_and_the_component() {
  let error =
    normalise_row(&[1.0, f32::NAN, 2.0], 5, space(3).space()).expect_err("NaN is not an embedding");
  assert!(
    matches!(error, Error::NonFiniteOutput(payload) if payload.row() == 5 && payload.component() == 1),
    "expected NonFiniteOutput(5, 1), got {error:?}"
  );
}

#[test]
fn cosine_is_the_dot_product_of_unit_vectors() {
  let a = normalise_row(&[1.0, 0.0], 0, space(2).space()).expect("unit");
  let b = normalise_row(&[0.0, 1.0], 0, space(2).space()).expect("unit");
  let c = normalise_row(&[1.0, 0.0], 0, space(2).space()).expect("unit");
  assert!(a.cosine(&b).expect("one space").abs() < 1e-6);
  assert!((a.cosine(&c).expect("one space") - 1.0).abs() < 1e-6);
  assert_eq!(
    a.dot(&b).expect("one space"),
    a.cosine(&b).expect("one space")
  );
}

#[test]
fn one_space_reached_through_two_separately_built_manifests_still_compares() {
  // `&self` inference means fan-out is one embedder per worker over the same
  // artifact, so the SAME space is legitimately produced by more than one
  // producer. A space identity minted per embedder would refuse exactly the
  // cross-worker comparisons those workers exist to make; this pins that it
  // does not. See `FaceEmbedding`'s doc for why the precedent's minted token
  // was not the right shape here.
  let a = normalise_row(&[1.0, 0.0], 0, space(2).space()).expect("unit");
  let b = normalise_row(
    &[1.0, 0.0],
    0,
    FaceModel::new("data", "embedding", 2).space(),
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
  let a = normalise_row(&[1.0, 0.0], 0, space(2).space()).expect("unit");
  let wide = normalise_row(&[1.0, 0.0, 0.0], 0, space(3).space()).expect("unit");
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
  let rgb = space(2);
  let bgr = rgb.with_preprocessing(Preprocessing::from_mean_and_divisor(
    ChannelOrder::Bgr,
    TensorLayout::Nchw,
    [127.5, 127.5, 127.5],
    127.5,
  ));
  let a = normalise_row(&[1.0, 0.0], 0, rgb.space()).expect("unit");
  let b = normalise_row(&[1.0, 0.0], 0, bgr.space()).expect("unit");
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
    let far = normalise_row(&[1.0, 0.0], 0, other.space()).expect("unit");
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
  // either embedding. A broken manifest is one broken manifest.
  let with_nan = |payload: f32| {
    space(2).with_preprocessing(Preprocessing::new(
      ChannelOrder::Rgb,
      TensorLayout::Nchw,
      payload,
      [payload, -1.0, -1.0],
    ))
  };
  let nan = with_nan(f32::NAN);
  let a = normalise_row(&[1.0, 0.0], 0, nan.space()).expect("unit");
  // Deliberately a DIFFERENT NaN payload, not a second copy of the same
  // constant: a raw `to_bits` comparison passes the same-constant case and
  // fails this one, so the same-constant case alone cannot tell a canonical
  // relation from a bitwise one. `f32` has 2²³ − 1 quiet NaN payloads and an
  // arithmetic result may be any of them.
  let other_payload = f32::from_bits(f32::NAN.to_bits() | 1);
  assert!(other_payload.is_nan() && other_payload.to_bits() != f32::NAN.to_bits());
  let b = normalise_row(&[1.0, 0.0], 0, with_nan(other_payload).space()).expect("unit");
  assert_eq!(
    a.cosine(&b)
      .expect("one manifest is one space, whatever it holds"),
    1.0
  );
  assert!(
    a.cosine(&normalise_row(&[1.0, 0.0], 0, space(2).space()).expect("unit"))
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
  let manifest = space(2).with_preprocessing(Preprocessing::from_mean_and_divisor(
    ChannelOrder::Bgr,
    TensorLayout::Nhwc,
    [104.0, 117.0, 123.0],
    58.0,
  ));
  assert_ne!(manifest.preprocessing(), Preprocessing::ARCFACE);
  let embedding = normalise_row(&[1.0, 0.0], 0, manifest.space()).expect("unit");
  assert_eq!(embedding.space(), manifest.space());
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
    let contract = resolve_input_contract(&declared, F32, layout)
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
  let error = check_predicted_shape(&[512, 4], 4 * 512, OutputContract::Batched, 4, 512)
    .expect_err("a [dim, batch] tensor is not a [batch, dim] tensor");
  assert!(
    matches!(&error, Error::OutputShape(payload) if payload.got() == [512, 4]),
    "expected OutputShape([512, 4]), got {error:?}"
  );

  // The contract's own shape still passes, and so does the batch-one `[dim]`
  // form — but ONLY for a batch-one contract.
  assert!(check_predicted_shape(&[4, 512], 4 * 512, OutputContract::Batched, 4, 512).is_ok());
  assert!(check_predicted_shape(&[512], 512, OutputContract::Flat, 1, 512).is_ok());

  // A graph that declared nothing may emit either form — and only those two.
  assert!(check_predicted_shape(&[1, 512], 512, OutputContract::Undeclared, 1, 512).is_ok());
  assert!(check_predicted_shape(&[512], 512, OutputContract::Undeclared, 1, 512).is_ok());
  assert!(
    check_predicted_shape(&[512], 512, OutputContract::Undeclared, 512, 1).is_err(),
    "a bare [dim] tensor must not stand in for a 512-row batch"
  );
  assert!(
    check_predicted_shape(&[512, 4], 4 * 512, OutputContract::Undeclared, 4, 512).is_err(),
    "an undeclared output shape must not become a licence to accept any transposition"
  );

  // And a declared form is binding: a graph that promised [batch, dim] does
  // not get to emit [dim] instead.
  assert!(check_predicted_shape(&[512], 512, OutputContract::Batched, 1, 512).is_err());
}

#[test]
fn a_diverging_element_count_is_not_reported_as_a_shape_mismatch() {
  // FALSIFIER (red first, on CONTENTS). `count` is CoreML's OWN answer rather
  // than a product of the cached shape — which is the entire reason it is
  // checked alongside the axes. So the two can disagree, and when only the
  // COUNT does, the shape matched: the old payload put the same vector in
  // both fields and rendered "expected [4, 512], got [4, 512]", a shape
  // mismatch that did not happen.
  let error = check_predicted_shape(&[4, 512], 4 * 512 - 1, OutputContract::Batched, 4, 512)
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
      check_predicted_shape(&[512, 4], 4 * 512, OutputContract::Batched, 4, 512),
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
  let big = normalise_row(&[1.8e38, 2.4e38], 0, space(2).space())
    .expect("a large but finite row has a direction");
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

  let small = normalise_row(&[3.0e-25, 4.0e-25], 1, space(2).space())
    .expect("a tiny but finite row has a direction");
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
    normalise_row(&[0.0, -0.0], 2, space(2).space()).expect_err("zero has no direction"),
    Error::EmbeddingZero(_)
  ));
}

#[test]
fn the_load_time_contract_requires_an_f32_multi_array() {
  // Names and shapes were checked; the tensor KIND and element type never
  // were, while inference always supplies and extracts `f32` multi-arrays. An
  // f16 export therefore loaded clean and failed every prediction.
  for wrong in [DataType::F16, DataType::F64, DataType::I32] {
    assert_eq!(
      resolve_input_contract(
        &[1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE],
        Some(wrong),
        TensorLayout::Nchw
      ),
      None,
      "a {wrong} input must not load against an f32 inference path"
    );
    assert!(
      check_output_contract(&[4, 512], Some(wrong), 4, 512).is_err(),
      "a {wrong} output must not load against an f32 inference path"
    );
  }

  // `data_type()` is `None` exactly when the feature is NOT a multi-array —
  // which is the case the module doc's own census hits: both third-party
  // CoreML ArcFace builds it surveys declare `ImageType` inputs, and an image
  // feature reports no shape and no dtype at all.
  assert_eq!(
    resolve_input_contract(&[], None, TensorLayout::Nchw),
    None,
    "a feature that is not a multi-array must not resolve to batch 1"
  );
  assert!(
    check_output_contract(&[], None, 1, 512).is_err(),
    "a feature that is not a multi-array must not satisfy the output contract"
  );

  // An f32 multi-array still loads, including the undeclared-shape form a
  // legacy `neuralNetwork` artifact leaves behind.
  assert_eq!(
    contract_of(&[1, 3, TEMPLATE_SIZE, TEMPLATE_SIZE], TensorLayout::Nchw),
    Some((1, InputRank::Batched))
  );
  assert_eq!(
    contract_of(&[], TensorLayout::Nchw),
    Some((1, InputRank::Batched))
  );
  assert!(check_output_contract(&[4, 512], F32, 4, 512).is_ok());
  assert!(check_output_contract(&[], F32, 4, 512).is_ok());
}

#[test]
fn manifest_equality_and_space_identity_are_one_relation() {
  // `FaceModel` used to derive `PartialEq` — `f32`'s `==`, under which `+0.0`
  // and `−0.0` are equal — while `space_difference` compared the same `f32`s
  // by RAW BIT PATTERN, under which they are not. One type, two equality
  // relations, contradicting each other; and it was the space check that got
  // it wrong, refusing a pair whose preprocessing is the same function.
  let plus = space(2).with_preprocessing(Preprocessing::new(
    ChannelOrder::Rgb,
    TensorLayout::Nchw,
    1.0 / 127.5,
    [0.0, -1.0, -1.0],
  ));
  let minus = space(2).with_preprocessing(Preprocessing::new(
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

  let a = normalise_row(&[1.0, 0.0], 0, plus.space()).expect("unit");
  let b = normalise_row(&[1.0, 0.0], 0, minus.space()).expect("unit");
  for (name, related) in [
    ("FaceModel: PartialEq", plus == minus),
    (
      "Preprocessing: PartialEq",
      plus.preprocessing() == minus.preprocessing(),
    ),
    ("EmbeddingSpace: PartialEq", plus.space() == minus.space()),
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
  let other = space(2).with_preprocessing(Preprocessing::new(
    ChannelOrder::Rgb,
    TensorLayout::Nchw,
    1.0 / 127.5,
    [1e-45, -1.0, -1.0],
  ));
  assert_ne!(plus.space(), other.space(), "a subnormal bias is not zero");
  assert!(
    a.dot(&normalise_row(&[1.0, 0.0], 0, other.space()).expect("unit"))
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
    model.space().hash(&mut hasher);
    hasher.finish()
  };
  let with = |bias: [f32; 3], scale: f32| {
    space(2).with_preprocessing(Preprocessing::new(
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
    assert_eq!(left.space(), right.space(), "these name one space");
    assert_eq!(
      hash_of(left),
      hash_of(right),
      "equal spaces must hash equal, or a map keyed by one is broken"
    );
  }
}

#[test]
fn feature_names_are_io_routing_and_do_not_decide_the_space() {
  // A feature name is the string CoreML routes a tensor by. Renaming a
  // graph's input and output re-exports the SAME weights, so the vectors stay
  // in one space; and two unrelated artifacts are free to use the same two
  // names. The name is therefore neither necessary nor sufficient evidence
  // about the space, and comparing it only produces false refusals.
  let renamed = FaceModel::new("input_1", "var_2011", 2);
  let original = space(2);
  let a = normalise_row(&[1.0, 0.0], 0, original.space()).expect("unit");
  let b = normalise_row(&[1.0, 0.0], 0, renamed.space()).expect("unit");
  assert_eq!(
    a.dot(&b)
      .expect("the same weights re-exported under other feature names is one space"),
    1.0
  );
}

#[test]
fn the_space_a_vector_carries_is_one_its_caller_built() {
  // Pinned because a previous round asserted the opposite — that no public
  // `FaceModel` constructor exists — and rested the whole space guarantee on
  // it. `FaceModel::new` and `Preprocessing::new` are both public `const fn`
  // and `FaceEmbedder::load` takes the manifest from its caller, so every
  // field of every stamped space is a value the caller chose. What a caller
  // cannot do is assemble the VECTOR: that is where the guarantee actually
  // lives, and this gate is here so the false version cannot be written back.
  let caller_built = FaceModel::new("data", "embedding", 2).with_preprocessing(Preprocessing::new(
    ChannelOrder::Bgr,
    TensorLayout::Nhwc,
    1.0 / 128.0,
    [-1.0, -1.0, -1.0],
  ));
  let embedding = normalise_row(&[1.0, 0.0], 0, caller_built.space()).expect("unit");
  assert_eq!(
    embedding.space(),
    caller_built.space(),
    "the stamped space is exactly the one the caller handed to `load`"
  );

  // The residual, as an assertion rather than a paragraph: two DISTINCT
  // artifacts declaring the same width and preprocessing are one space as far
  // as this crate can see, and their cosine is returned rather than refused.
  // Nothing short of holding the weights closes it.
  let second_artifact =
    FaceModel::new("input_1", "var_9", 2).with_preprocessing(Preprocessing::new(
      ChannelOrder::Bgr,
      TensorLayout::Nhwc,
      1.0 / 128.0,
      [-1.0, -1.0, -1.0],
    ));
  let other = normalise_row(&[1.0, 0.0], 0, second_artifact.space()).expect("unit");
  assert_eq!(
    embedding
      .dot(&other)
      .expect("one space, as far as anything here can tell"),
    1.0,
    "the residual: schema equality is not artifact identity, and this crate holds no weights"
  );
}
