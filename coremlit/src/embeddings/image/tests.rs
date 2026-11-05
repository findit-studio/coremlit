use super::*;

#[test]
fn accepts_exactly_width_times_height_times_three() {
  let data = vec![7u8; 4 * 3 * 3];
  let img = Rgb8Image::new(&data, 4, 3).expect("valid");
  assert_eq!((img.width(), img.height()), (4, 3));
  assert_eq!(img.data(), data.as_slice());

  // One byte either side of the exact length is a length refusal.
  for len in [4 * 3 * 3 - 1, 4 * 3 * 3 + 1] {
    let bad = vec![0u8; len];
    assert!(matches!(
      Rgb8Image::new(&bad, 4, 3),
      Err(ImageGeometryError::ImageDataLength(_))
    ));
  }
}

#[test]
fn a_zero_side_is_an_image_dimensions_refusal_carrying_both_sides() {
  let empty: Vec<u8> = Vec::new();
  match Rgb8Image::new(&empty, 0, 3) {
    Err(ImageGeometryError::ImageDimensions(e)) => assert_eq!((e.width(), e.height()), (0, 3)),
    other => panic!("expected ImageDimensions, got {other:?}"),
  }
  match Rgb8Image::new(&empty, 4, 0) {
    Err(ImageGeometryError::ImageDimensions(e)) => assert_eq!((e.width(), e.height()), (4, 0)),
    other => panic!("expected ImageDimensions, got {other:?}"),
  }
}

#[test]
fn an_overflowing_product_is_an_image_dimensions_refusal() {
  // The data length is irrelevant to the overflow.
  let data = [0u8; 1];
  for (w, h) in [(usize::MAX, 2), (usize::MAX / 3 + 1, 1), (2, usize::MAX)] {
    match Rgb8Image::new(&data, w, h) {
      Err(ImageGeometryError::ImageDimensions(e)) => assert_eq!((e.width(), e.height()), (w, h)),
      other => panic!("expected ImageDimensions for {w}x{h}, got {other:?}"),
    }
  }
}

#[test]
fn a_wrong_length_is_an_image_data_length_refusal_carrying_got_and_expected() {
  let data = vec![0u8; 4 * 3 * 3 - 1];
  match Rgb8Image::new(&data, 4, 3) {
    Err(ImageGeometryError::ImageDataLength(e)) => {
      assert_eq!(e.got(), 4 * 3 * 3 - 1);
      assert_eq!(e.expected(), 4 * 3 * 3);
    }
    other => panic!("expected ImageDataLength, got {other:?}"),
  }
}

#[test]
fn the_geometry_error_messages_are_pinned() {
  assert_eq!(
    ImageGeometryError::ImageDimensions(ImageDimensions::new(640, 0)).to_string(),
    "invalid image dimensions: 640×0 (zero, over the per-axis maximum, or size overflow)"
  );
  assert_eq!(
    ImageGeometryError::ImageDataLength(ImageDataLength::new(100, 921_600)).to_string(),
    "image data length mismatch: expected 921600 bytes (w·h·3), got 100"
  );
}

#[cfg(feature = "siglip")]
#[test]
fn the_siglip_error_texts_are_the_pre_lift_texts() {
  use crate::embeddings::siglip::Error;

  let dims = Error::from(ImageGeometryError::ImageDimensions(ImageDimensions::new(
    640, 0,
  )));
  assert_eq!(
    dims.to_string(),
    "invalid image dimensions: 640×0 (zero, over the per-axis maximum, or size overflow)"
  );
  let len = Error::from(ImageGeometryError::ImageDataLength(ImageDataLength::new(
    100, 921_600,
  )));
  assert_eq!(
    len.to_string(),
    "image data length mismatch: expected 921600 bytes (w·h·3), got 100"
  );
}

#[cfg(feature = "siglip")]
#[test]
fn the_conversion_keeps_each_variant_and_payload() {
  use crate::embeddings::siglip::Error;

  match Error::from(ImageGeometryError::ImageDimensions(ImageDimensions::new(
    3, 0,
  ))) {
    Error::ImageDimensions(e) => assert_eq!((e.width(), e.height()), (3, 0)),
    other => panic!("expected ImageDimensions, got {other:?}"),
  }
  match Error::from(ImageGeometryError::ImageDataLength(ImageDataLength::new(
    5, 9,
  ))) {
    Error::ImageDataLength(e) => assert_eq!((e.got(), e.expected()), (5, 9)),
    other => panic!("expected ImageDataLength, got {other:?}"),
  }
}

/// The shared types keep resolving at the siglip paths they were first public
/// at, and every path names the one type.
#[cfg(feature = "siglip")]
#[test]
fn the_siglip_paths_resolve_to_the_shared_types() {
  use crate::embeddings::siglip::{
    self,
    error::{
      ImageDataLength as LengthViaError, ImageDimensions as DimensionsViaError,
      ImageGeometryError as GeometryViaError,
    },
    image::Rgb8Image as ViaImage,
  };

  let data = [0u8; 12];
  let via_image: ViaImage<'_> = ViaImage::new(&data, 2, 2).expect("valid");
  let via_root: siglip::Rgb8Image<'_> = siglip::Rgb8Image::new(&data, 2, 2).expect("valid");
  let shared: Rgb8Image<'_> = Rgb8Image::new(&data, 2, 2).expect("valid");
  assert_eq!(via_image.data(), shared.data());
  assert_eq!(via_root.data(), shared.data());

  let dims = DimensionsViaError::new(1, 2);
  let len = LengthViaError::new(1, 2);
  let geometry: GeometryViaError = GeometryViaError::ImageDimensions(dims);
  let via_root: siglip::ImageGeometryError = siglip::ImageGeometryError::ImageDataLength(len);
  assert!(matches!(geometry, ImageGeometryError::ImageDimensions(_)));
  assert!(matches!(via_root, ImageGeometryError::ImageDataLength(_)));
  assert_eq!(siglip::ImageDimensions::new(4, 5).width(), 4);
  assert_eq!(siglip::ImageDataLength::new(6, 7).expected(), 7);
}

/// The lines of the pre-lift golden that start with one of `prefixes`, in file
/// order. The golden's `# ` lines are commentary.
#[cfg(feature = "siglip")]
fn golden_lines(prefixes: &[&str]) -> Vec<&'static str> {
  include_str!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/tests/fixtures/lifted_observables_812da2d.txt"
  ))
  .lines()
  .filter(|line| !line.starts_with("# "))
  .filter(|line| prefixes.iter().any(|prefix| line.starts_with(prefix)))
  .collect()
}

/// Every mismatch between a rendering and its golden lines, so a failure names
/// all the moved observables at once.
#[cfg(feature = "siglip")]
fn mismatches(rendered: &[String], golden: &[&str]) -> Vec<String> {
  let mut out = Vec::new();
  if rendered.len() != golden.len() {
    out.push(format!(
      "line count: rendered {}, golden {}",
      rendered.len(),
      golden.len()
    ));
  }
  for (index, (got, want)) in rendered.iter().zip(golden).enumerate() {
    if got != want {
      out.push(format!(
        "line {index}\n  rendered: {got}\n  golden:   {want}"
      ));
    }
  }
  out
}

/// One buffer through both siglip paths of `Rgb8Image::new`, rendered like the
/// golden's lines.
#[cfg(feature = "siglip")]
fn render_rgb8(out: &mut Vec<String>, label: &str, len: usize, w: usize, h: usize) {
  use crate::embeddings::siglip;

  let data = vec![0u8; len];
  for (path, result) in [
    (
      "siglip::Rgb8Image",
      siglip::Rgb8Image::new(&data, w, h).map(|_| ()),
    ),
    (
      "siglip::image::Rgb8Image",
      siglip::image::Rgb8Image::new(&data, w, h).map(|_| ()),
    ),
  ] {
    match result {
      Ok(()) => out.push(format!("{path} {label}: ok")),
      Err(e) => out.push(format!(
        "{path} {label}: display={e} | debug={e:?} | source={:?}",
        std::error::Error::source(&e).map(|s| s.to_string())
      )),
    }
  }
}

/// Every observable the image lift moved is what the build before the lift
/// printed: the refusal text and `Debug` of each `Rgb8Image::new` outcome on
/// both siglip paths, siglip's own `Error` texts and the payloads' `Debug`.
#[cfg(feature = "siglip")]
#[test]
fn the_image_observables_are_the_pre_lift_observables() {
  use crate::embeddings::siglip;

  let mut rendered = Vec::new();
  render_rgb8(&mut rendered, "zero width", 0, 0, 3);
  render_rgb8(&mut rendered, "zero height", 0, 4, 0);
  render_rgb8(&mut rendered, "overflow usize::MAX x 2", 1, usize::MAX, 2);
  render_rgb8(
    &mut rendered,
    "overflow (usize::MAX/3+1) x 1",
    1,
    usize::MAX / 3 + 1,
    1,
  );
  render_rgb8(&mut rendered, "short 35 for 4x3", 35, 4, 3);
  render_rgb8(&mut rendered, "long 37 for 4x3", 37, 4, 3);
  render_rgb8(&mut rendered, "exact 36 for 4x3", 36, 4, 3);
  rendered.push(format!(
    "siglip::Error::ImageDimensions: {}",
    siglip::Error::ImageDimensions(siglip::ImageDimensions::new(640, 0))
  ));
  rendered.push(format!(
    "siglip::Error::ImageDataLength: {}",
    siglip::Error::ImageDataLength(siglip::error::ImageDataLength::new(100, 921_600))
  ));
  rendered.push(format!(
    "payload debug: {:?} {:?}",
    siglip::error::ImageDimensions::new(1, 2),
    siglip::ImageDataLength::new(3, 4)
  ));

  let golden = golden_lines(&[
    "siglip::Rgb8Image ",
    "siglip::image::Rgb8Image ",
    "siglip::Error::",
    "payload debug: ImageDimensions",
  ]);
  assert_eq!(golden.len(), 17, "the image law covers 17 golden lines");
  let moved = mismatches(&rendered, &golden);
  assert!(
    moved.is_empty(),
    "{} observable(s) differ from the pre-lift golden:\n{}",
    moved.len(),
    moved.join("\n")
  );
}
