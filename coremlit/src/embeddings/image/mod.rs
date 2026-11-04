//! The decoded-image view the image doors share: the borrowed [`Rgb8Image`], the
//! [`ImageGeometryError`] its constructor refuses with, and the two payloads
//! that error carries.
//!
//! The view is the sans-I/O seam of an image door: decoding PNG/JPEG is the
//! caller's job, so the crate gains no image-decoder runtime dependency. Each
//! consuming door re-exports these types at its own path and converts
//! [`ImageGeometryError`] into its own `Error`, variant for variant.

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
  /// [`ImageGeometryError::ImageDimensions`] if `width` or `height` is zero, or
  /// if `width · height · 3` overflows `usize`;
  /// [`ImageGeometryError::ImageDataLength`] if `data.len()` is not exactly
  /// `width · height · 3`.
  pub fn new(data: &'a [u8], width: usize, height: usize) -> Result<Self, ImageGeometryError> {
    if width == 0 || height == 0 {
      return Err(ImageGeometryError::ImageDimensions(ImageDimensions::new(
        width, height,
      )));
    }
    let expected = width
      .checked_mul(height)
      .and_then(|hw| hw.checked_mul(3))
      .ok_or(ImageGeometryError::ImageDimensions(ImageDimensions::new(
        width, height,
      )))?;
    if data.len() != expected {
      return Err(ImageGeometryError::ImageDataLength(ImageDataLength::new(
        data.len(),
        expected,
      )));
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

/// Why [`Rgb8Image::new`] refused a buffer: its geometry is not a valid
/// `width · height · 3` RGB8 image.
///
/// A door's own error type converts this variant for variant, so a caller
/// matching on the door's `Error` sees the same cause under the same name.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ImageGeometryError {
  /// [`Rgb8Image::new`] refused a zero side or an overflowing
  /// `width · height · 3`.
  ///
  /// The text is the one this refusal has always printed; it also names the
  /// per-axis maximum a door's own preprocessing refuses under the same
  /// payload (siglip's, for one).
  #[error(
    "invalid image dimensions: {}×{} (zero, over the per-axis maximum, or size overflow)",
    .0.width(),
    .0.height()
  )]
  ImageDimensions(ImageDimensions),

  /// The backing slice length was not `width · height · 3`.
  #[error("image data length mismatch: expected {} bytes (w·h·3), got {}", .0.expected(), .0.got())]
  ImageDataLength(ImageDataLength),
}

/// An [`Rgb8Image`] view had a zero dimension or a `width · height · 3` byte
/// length overflowing `usize`. A real decoded RGB image has non-zero dimensions
/// and a length that fits.
///
/// Payload of [`ImageGeometryError::ImageDimensions`]. A door's own dimension
/// refusal may reuse it for further bounds of its own (siglip's also covers an
/// axis over its preprocessing bound).
#[derive(Debug)]
pub struct ImageDimensions {
  /// The width supplied.
  width: usize,
  /// The height supplied.
  height: usize,
}

impl ImageDimensions {
  /// Construct from the width and the height supplied.
  #[inline(always)]
  pub const fn new(width: usize, height: usize) -> Self {
    Self { width, height }
  }

  /// The width supplied.
  #[inline(always)]
  pub const fn width(&self) -> usize {
    self.width
  }

  /// The height supplied.
  #[inline(always)]
  pub const fn height(&self) -> usize {
    self.height
  }
}

/// An [`Rgb8Image`] view's backing slice length did not equal
/// `width · height · 3` (row-major, RGB-interleaved, 3 bytes/pixel).
///
/// Payload of [`ImageGeometryError::ImageDataLength`].
#[derive(Debug)]
pub struct ImageDataLength {
  /// The backing slice length the caller supplied.
  got: usize,
  /// The required `width · height · 3` length.
  expected: usize,
}

impl ImageDataLength {
  /// Construct from the backing slice length the caller supplied and the
  /// required `width · height · 3` length.
  #[inline(always)]
  pub const fn new(got: usize, expected: usize) -> Self {
    Self { got, expected }
  }

  /// The backing slice length the caller supplied.
  #[inline(always)]
  pub const fn got(&self) -> usize {
    self.got
  }

  /// The required `width · height · 3` length.
  #[inline(always)]
  pub const fn expected(&self) -> usize {
    self.expected
  }
}

#[cfg(test)]
mod tests;
