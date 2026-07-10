//! Owned, typed N-dimensional CoreML arrays.

use half::f16;
use objc2::{AnyThread, rc::Retained};
use objc2_core_ml::{MLMultiArray, MLMultiArrayDataType};
use objc2_foundation::{NSArray, NSNumber};

use crate::{DataType, NsErrorInfo, ShapeRequirement, TensorError};

mod sealed {
  pub trait Sealed {}
  impl Sealed for half::f16 {}
  impl Sealed for f32 {}
  impl Sealed for f64 {}
  impl Sealed for i32 {}
}

/// Element types storable in a [`MultiArray`].
pub trait Element: sealed::Sealed + Copy {
  /// The corresponding CoreML data type.
  const DATA_TYPE: DataType;
}

impl Element for f16 {
  const DATA_TYPE: DataType = DataType::F16;
}
impl Element for f32 {
  const DATA_TYPE: DataType = DataType::F32;
}
impl Element for f64 {
  const DATA_TYPE: DataType = DataType::F64;
}
impl Element for i32 {
  const DATA_TYPE: DataType = DataType::I32;
}

/// An owned CoreML `MLMultiArray` with typed element access.
///
/// Mutation requires `&mut self`; typed views check the element type at
/// runtime and never reinterpret bytes across types.
#[derive(Debug)]
pub struct MultiArray {
  inner: Retained<MLMultiArray>,
}

// SAFETY: MLMultiArray owns its buffer; ownership transfer across threads is
// sound. Not `Sync`: unsynchronized `&self` reads concurrent with FFI-side
// mutation are not proven safe.
unsafe impl Send for MultiArray {}

fn ns_shape(shape: &[usize]) -> Retained<NSArray<NSNumber>> {
  let numbers: Vec<Retained<NSNumber>> = shape.iter().map(|d| NSNumber::new_usize(*d)).collect();
  NSArray::from_retained_slice(&numbers)
}

impl MultiArray {
  /// Allocates an array of `shape` and fills it with zero bytes.
  ///
  /// # Errors
  /// [`TensorError::UnsupportedDataType`] if `dtype` has no known element
  /// size (e.g. an unrecognized [`DataType::Unknown`] code); CoreML's
  /// `initWithShape_dataType_error` does not reject such codes itself, and
  /// an unsized dtype would make the zero-fill below unsound.
  /// [`TensorError::Native`] if CoreML rejects the allocation.
  pub fn zeros(shape: &[usize], dtype: DataType) -> Result<Self, TensorError> {
    if dtype.size_of().is_none() {
      return Err(TensorError::UnsupportedDataType { dtype });
    }
    // SAFETY: valid shape array; `dtype` was checked above to have a known
    // element size, so even though CoreML does not itself validate the
    // data-type code, the buffer this allocates is one this crate knows how
    // to size and zero-fill. Result checked.
    let inner = unsafe {
      MLMultiArray::initWithShape_dataType_error(
        MLMultiArray::alloc(),
        &ns_shape(shape),
        MLMultiArrayDataType(dtype.to_raw()),
      )
    }
    .map_err(|e| TensorError::Native(NsErrorInfo::from_ns_error(&e)))?;
    let mut this = Self { inner };
    // MLMultiArray does not guarantee zeroed memory; mirror ArgmaxCore's
    // explicit initialValue fill.
    this.fill_bytes_zero();
    Ok(this)
  }

  /// Builds an array of `shape` from `data` (row-major).
  ///
  /// # Errors
  /// [`TensorError::ShapeMismatch`] if `data.len()` differs from the shape
  /// product; [`TensorError::Native`] if allocation fails.
  pub fn from_slice<T>(shape: &[usize], data: &[T]) -> Result<Self, TensorError>
  where
    T: Element,
  {
    let expected: usize = shape.iter().product();
    if expected != data.len() {
      return Err(TensorError::ShapeMismatch {
        expected,
        actual: data.len(),
      });
    }
    let mut this = Self::zeros(shape, T::DATA_TYPE)?;
    this.as_slice_mut::<T>()?.copy_from_slice(data);
    Ok(this)
  }

  /// The dimension sizes.
  pub fn shape(&self) -> Vec<usize> {
    // SAFETY: accessor message send on a live object.
    unsafe { self.inner.shape() }
      .iter()
      .map(|n| n.as_usize())
      .collect()
  }

  /// The stride, in elements, of each dimension.
  pub fn strides(&self) -> Vec<usize> {
    // SAFETY: accessor message send on a live object.
    unsafe { self.inner.strides() }
      .iter()
      .map(|n| n.as_usize())
      .collect()
  }

  /// Whether the elements are laid out contiguously in row-major order.
  ///
  /// Arrays from [`Self::zeros`]/[`Self::from_slice`] always are. Pixel-
  /// buffer-backed arrays ([`Self::f16_surface`]) may carry row padding,
  /// which surfaces here as non-default strides.
  pub fn is_contiguous(&self) -> bool {
    let shape = self.shape();
    let strides = self.strides();
    let mut expected = 1usize;
    // Walk dims minor→major comparing against canonical row-major strides.
    for (dim, stride) in shape.iter().zip(&strides).rev() {
      if *stride != expected {
        return false;
      }
      expected *= (*dim).max(1);
    }
    true
  }

  /// Total number of elements.
  pub fn count(&self) -> usize {
    // SAFETY: accessor message send on a live object.
    unsafe { self.inner.count() as usize }
  }

  /// The element type.
  pub fn data_type(&self) -> DataType {
    // SAFETY: accessor message send on a live object.
    DataType::from_raw(unsafe { self.inner.dataType() }.0)
  }

  /// Borrows the elements as `&[T]`.
  ///
  /// # Errors
  /// [`TensorError::DataTypeMismatch`] if `T` differs from
  /// [`Self::data_type`]. [`TensorError::NonContiguous`] if the array's
  /// memory layout carries padding (see [`Self::is_contiguous`]) — a flat
  /// slice cannot represent gaps between rows, so it is refused rather than
  /// silently exposing/hiding padding bytes.
  pub fn as_slice<T>(&self) -> Result<&[T], TensorError>
  where
    T: Element,
  {
    self.check_dtype::<T>()?;
    if !self.is_contiguous() {
      return Err(TensorError::NonContiguous {
        shape: self.shape(),
        strides: self.strides(),
      });
    }
    // SAFETY: dtype checked; contiguity checked above, so the flat range
    // `[dataPointer, dataPointer + count * size_of::<T>())` holds exactly
    // `count()` densely packed elements with no row gaps; lifetime tied to
    // &self. `dataPointer` is the only stable way to hand out a borrow (the
    // block-based accessors scope the pointer to a closure); Swift
    // WhisperKit does the same.
    #[allow(deprecated)]
    Ok(unsafe {
      core::slice::from_raw_parts(self.inner.dataPointer().as_ptr().cast(), self.count())
    })
  }

  /// Borrows the elements as `&mut [T]`.
  ///
  /// # Errors
  /// [`TensorError::DataTypeMismatch`] if `T` differs from
  /// [`Self::data_type`]. [`TensorError::NonContiguous`] as in
  /// [`Self::as_slice`].
  pub fn as_slice_mut<T>(&mut self) -> Result<&mut [T], TensorError>
  where
    T: Element,
  {
    self.check_dtype::<T>()?;
    if !self.is_contiguous() {
      return Err(TensorError::NonContiguous {
        shape: self.shape(),
        strides: self.strides(),
      });
    }
    // SAFETY: as in `as_slice` (dtype and contiguity both checked above),
    // plus exclusivity via &mut self.
    #[allow(deprecated)]
    Ok(unsafe {
      core::slice::from_raw_parts_mut(self.inner.dataPointer().as_ptr().cast(), self.count())
    })
  }

  /// Linear element offset of an index tuple, honoring strides.
  ///
  /// # Errors
  /// [`TensorError::RankMismatch`] / [`TensorError::IndexOutOfBounds`].
  pub fn linear_offset(&self, indices: &[usize]) -> Result<usize, TensorError> {
    let shape = self.shape();
    if indices.len() != shape.len() {
      return Err(TensorError::RankMismatch {
        expected: shape.len(),
        actual: indices.len(),
      });
    }
    let strides = self.strides();
    let mut offset = 0usize;
    for ((&index, &dim), &stride) in indices.iter().zip(&shape).zip(&strides) {
      if index >= dim {
        return Err(TensorError::IndexOutOfBounds { index, len: dim });
      }
      offset += index * stride;
    }
    Ok(offset)
  }

  /// Writes `value` at one index tuple.
  ///
  /// # Errors
  /// Propagates [`Self::linear_offset`] and dtype-mismatch failures.
  pub fn fill_at<T>(&mut self, indices: &[usize], value: T) -> Result<(), TensorError>
  where
    T: Element,
  {
    let offset = self.linear_offset(indices)?;
    self.write_element(offset, value)
  }

  /// Writes `value` at each `position` of the final axis.
  ///
  /// Mirrors Swift's `fillLastDimension(indexes:with:)`, which
  /// preconditions a `[1, 1, n]` shape; this port generalizes that to any
  /// rank where every dimension before the last is 1 (rank 0/1 arrays have
  /// no leading dimensions and are always accepted here), so a genuinely
  /// multi-row array is rejected instead of only writing its first row.
  ///
  /// # Errors
  /// [`TensorError::UnsupportedShape`] (reason
  /// [`ShapeRequirement::LeadingDimsUnit`]) if any dimension before the
  /// last is not 1. [`TensorError::IndexOutOfBounds`] if a position exceeds
  /// the final axis — every position is validated before any element is
  /// written, so a single out-of-bounds position leaves the array
  /// untouched rather than partially filled. Dtype-mismatch failures from
  /// the typed view.
  pub fn fill_last_dim<T>(&mut self, positions: &[usize], value: T) -> Result<(), TensorError>
  where
    T: Element,
  {
    let shape = self.shape();
    if shape.len() >= 2 && shape[..shape.len() - 1].iter().any(|&dim| dim != 1) {
      return Err(TensorError::UnsupportedShape {
        shape,
        reason: ShapeRequirement::LeadingDimsUnit,
      });
    }
    let last = shape.last().copied().unwrap_or(0);
    let stride = self.strides().last().copied().unwrap_or(1);
    for &position in positions {
      if position >= last {
        return Err(TensorError::IndexOutOfBounds {
          index: position,
          len: last,
        });
      }
    }
    for &position in positions {
      self.write_element(position * stride, value)?;
    }
    Ok(())
  }

  /// Reads one element at an index tuple, honoring strides.
  ///
  /// Works on non-contiguous (row-padded) arrays that [`Self::as_slice`]
  /// refuses.
  ///
  /// # Errors
  /// A dtype mismatch is reported before any index validation; index
  /// bound/rank failures then propagate from [`Self::linear_offset`].
  /// (The write side validates in the opposite order — only the winning
  /// error differs when both conditions hold.)
  pub fn read_at<T>(&self, indices: &[usize]) -> Result<T, TensorError>
  where
    T: Element,
  {
    self.check_dtype::<T>()?;
    let offset = self.linear_offset(indices)?;
    // SAFETY: dtype checked above; `offset` is in-bounds because
    // `linear_offset` only returns successfully for indices already
    // validated against this array's own shape/strides — the read-side
    // mirror of `write_element`'s trust boundary.
    #[allow(deprecated)]
    let value = unsafe { *self.inner.dataPointer().as_ptr().cast::<T>().add(offset) };
    Ok(value)
  }

  /// Gathers the whole logical array into `out`, in row-major order,
  /// honoring strides.
  ///
  /// Unlike [`Self::as_slice`], this works on non-contiguous (row-padded)
  /// arrays: contiguous arrays are copied in one range read, and padded
  /// ones are gathered row by row (each row — everything but the last
  /// dimension — is copied as one contiguous run when the last dimension's
  /// stride is 1, which holds for every layout this crate produces; a
  /// per-element fallback covers the case where it doesn't).
  ///
  /// # Errors
  /// [`TensorError::ShapeMismatch`] if `out.len() != self.count()`.
  /// Dtype-mismatch failures.
  pub fn copy_into<T>(&self, out: &mut [T]) -> Result<(), TensorError>
  where
    T: Element,
  {
    self.check_dtype::<T>()?;
    let count = self.count();
    if out.len() != count {
      return Err(TensorError::ShapeMismatch {
        expected: count,
        actual: out.len(),
      });
    }
    if self.is_contiguous() {
      // `as_slice` re-checks dtype (already confirmed above) and
      // contiguity (just confirmed here), so this cannot fail; reusing it
      // avoids duplicating the flat-read unsafe block.
      out.copy_from_slice(self.as_slice::<T>()?);
      return Ok(());
    }

    // Padded layout (e.g. `f16_surface`): CoreML only ever pads *between*
    // rows, never inside one, so a rank >= 1 array's last dimension is one
    // "row". `is_contiguous` returning false above guarantees rank >= 1
    // here (a rank-0 shape trivially satisfies `is_contiguous`), so
    // `shape[..rank - 1]` below never underflows.
    let shape = self.shape();
    let strides = self.strides();
    let rank = shape.len();
    let last_dim = shape[rank - 1];
    let last_stride = strides[rank - 1];
    let leading_dims = &shape[..rank - 1];
    let leading_strides = &strides[..rank - 1];
    let num_rows: usize = leading_dims.iter().product();

    let mut row_indices = vec![0usize; leading_dims.len()];
    for row in 0..num_rows {
      // Unravel `row` into a multi-index over `leading_dims`, row-major
      // (the last leading dimension varies fastest).
      let mut remainder = row;
      for i in (0..leading_dims.len()).rev() {
        row_indices[i] = remainder % leading_dims[i];
        remainder /= leading_dims[i];
      }
      let row_start: usize = row_indices
        .iter()
        .zip(leading_strides)
        .map(|(&index, &stride)| index * stride)
        .sum();
      let out_start = row * last_dim;

      if last_stride == 1 {
        // SAFETY: dtype checked above; `row_start` is a valid in-bounds
        // element offset built from this array's own shape/strides (each
        // `row_indices[i] < leading_dims[i]`), and the following
        // `last_dim` elements stay within that same row since its stride
        // is 1 — the same CoreML shape/strides trust boundary
        // `write_element` relies on, applied to a whole contiguous row at
        // once instead of one element.
        #[allow(deprecated)]
        let row_slice: &[T] = unsafe {
          core::slice::from_raw_parts(
            self.inner.dataPointer().as_ptr().cast::<T>().add(row_start),
            last_dim,
          )
        };
        out[out_start..out_start + last_dim].copy_from_slice(row_slice);
      } else {
        // Fallback: the last dimension itself is strided, so gather it one
        // element at a time.
        for last in 0..last_dim {
          // SAFETY: dtype checked above; `row_start + last * last_stride`
          // is a valid in-bounds element offset for the same reason as the
          // contiguous-row branch, one element at a time.
          #[allow(deprecated)]
          let value = unsafe {
            *self
              .inner
              .dataPointer()
              .as_ptr()
              .cast::<T>()
              .add(row_start + last * last_stride)
          };
          out[out_start + last] = value;
        }
      }
    }
    Ok(())
  }

  pub(crate) fn raw(&self) -> &MLMultiArray {
    &self.inner
  }

  // INVARIANT: callers must pass the sole `Retained` reference to this
  // array. `Send` and `as_slice_mut`'s exclusivity both assume no aliased
  // handle exists (Retained is Clone, so this cannot be enforced by type).
  pub(crate) fn from_raw(inner: Retained<MLMultiArray>) -> Self {
    Self { inner }
  }

  fn check_dtype<T>(&self) -> Result<(), TensorError>
  where
    T: Element,
  {
    let actual = self.data_type();
    if actual != T::DATA_TYPE {
      return Err(TensorError::DataTypeMismatch {
        expected: T::DATA_TYPE,
        actual,
      });
    }
    Ok(())
  }

  /// Writes `value` at a stride-derived element `offset`, bypassing
  /// [`Self::as_slice_mut`] so padded (non-contiguous) arrays stay writable
  /// through [`Self::fill_at`]/[`Self::fill_last_dim`].
  fn write_element<T>(&mut self, offset: usize, value: T) -> Result<(), TensorError>
  where
    T: Element,
  {
    self.check_dtype::<T>()?;
    // Bound: offset was computed from in-range indices against this
    // array's own strides, so it lies within the allocation.
    // SAFETY: dtype checked; exclusive access via &mut self; in-bounds per
    // the stride-derived offset argument above.
    #[allow(deprecated)]
    unsafe {
      *self.inner.dataPointer().as_ptr().cast::<T>().add(offset) = value;
    }
    Ok(())
  }

  fn fill_bytes_zero(&mut self) {
    let byte_len = self.count()
      * self
        .data_type()
        .size_of()
        .expect("constructors validate the data type");
    // SAFETY: `dataPointer` is non-null and suitably aligned for
    // `data_type()`, and the allocation backing it is at least
    // `count() * size_of(data_type())` bytes — guaranteed because every
    // constructor validates the dtype before the CoreML allocation call
    // (see `zeros`), so `byte_len` never exceeds the buffer. This method is
    // only ever called from `zeros`, whose `initWithShape_dataType_error`
    // allocation is always row-major contiguous (no padding), so a flat
    // `byte_len`-sized zero-fill covers exactly the array's elements with
    // no gaps to skip.
    #[allow(deprecated)]
    unsafe {
      core::ptr::write_bytes(self.inner.dataPointer().as_ptr().cast::<u8>(), 0, byte_len);
    }
  }
}

/// IOSurface-backed construction (ANE-efficient f16 I/O).
impl MultiArray {
  /// Allocates an IOSurface-backed half-precision array.
  ///
  /// CoreML shares IOSurface-backed `f16` buffers with the ANE without
  /// copies; WhisperKit allocates every f16 tensor this way. Width is the
  /// last dimension (minimum 1); height is `count / width`.
  ///
  /// The IOSurface may pad each row out to a platform-chosen alignment, so
  /// the result is not guaranteed [`Self::is_contiguous`]. When padded,
  /// bulk access via [`Self::as_slice`]/[`Self::as_slice_mut`] returns
  /// [`TensorError::NonContiguous`] rather than silently reading/writing
  /// through the padding; element access ([`Self::fill_at`],
  /// [`Self::fill_last_dim`]) and CoreML's own prediction APIs are
  /// stride-aware and read/write the correct bytes either way. Unlike
  /// [`Self::zeros`], the buffer's contents start uninitialized — callers
  /// must fill every element they read.
  ///
  /// # Errors
  /// [`TensorError::UnsupportedShape`] (reason
  /// [`ShapeRequirement::NonEmpty`]) if `shape` is empty — a rank-0 shape
  /// has no last dimension to serve as the pixel buffer width, so
  /// [`Self::shape`]/[`Self::linear_offset`] on the result would be
  /// nonsensical rather than merely unusual.
  /// [`TensorError::PixelBuffer`] if CoreVideo rejects the pixel buffer
  /// allocation (for example, a `0`-height buffer from an all-zero shape).
  pub fn f16_surface(shape: &[usize]) -> Result<Self, TensorError> {
    use objc2_core_foundation::{CFDictionary, CFRetained, CFType};
    use objc2_core_video::{
      CVPixelBuffer, CVPixelBufferCreate, kCVPixelBufferIOSurfacePropertiesKey,
      kCVPixelFormatType_OneComponent16Half, kCVReturnSuccess,
    };

    if shape.is_empty() {
      return Err(TensorError::UnsupportedShape {
        shape: Vec::new(),
        reason: ShapeRequirement::NonEmpty,
      });
    }

    let count: usize = shape.iter().product();
    let width = shape.last().copied().unwrap_or(1).max(1);
    let height = count / width;

    // An empty IOSurface-properties dictionary as the value of
    // `kCVPixelBufferIOSurfacePropertiesKey` opts the buffer into IOSurface
    // backing (mirrors Apple's own `initWithPixelBuffer:shape:` sample).
    let io_surface_props = CFDictionary::<CFType, CFType>::empty();
    let io_surface_props_ref: &CFType = io_surface_props.as_ref();
    // SAFETY: reads a linked, immutable `CFStringRef` constant exported by
    // the CoreVideo framework; the resulting `'static` reference is valid
    // for the lifetime of the process.
    let io_surface_key = unsafe { kCVPixelBufferIOSurfacePropertiesKey };
    let attrs = CFDictionary::from_slices(&[io_surface_key], &[io_surface_props_ref]);

    let mut pixel_buffer: *mut CVPixelBuffer = core::ptr::null_mut();
    // SAFETY: `width`/`height` are plain integers, `kCVPixelFormatType_OneComponent16Half`
    // is a real `OSType` constant, `attrs` is a live `CFDictionary` borrowed
    // for the duration of the call, and `pixel_buffer` is a valid, writable
    // local the out-pointer references. The `CVReturn` result is checked
    // immediately below before the (possibly still-null-on-failure)
    // out-pointer is used.
    let ret = unsafe {
      CVPixelBufferCreate(
        None,
        width,
        height,
        kCVPixelFormatType_OneComponent16Half,
        Some(attrs.as_opaque()),
        core::ptr::NonNull::from(&mut pixel_buffer),
      )
    };
    if ret != kCVReturnSuccess || pixel_buffer.is_null() {
      return Err(TensorError::PixelBuffer { code: ret });
    }
    let pixel_buffer =
      core::ptr::NonNull::new(pixel_buffer).expect("checked non-null CVPixelBuffer above");
    // SAFETY: `CVPixelBufferCreate` returned `kCVReturnSuccess` with a
    // non-null out-pointer, so `pixel_buffer` follows the Core Foundation
    // Create rule: a live object with a +1 retain count that this call
    // adopts into `CFRetained`, which will release it on drop.
    let buffer: CFRetained<CVPixelBuffer> = unsafe { CFRetained::from_raw(pixel_buffer) };

    // SAFETY: `buffer`'s pixel format is `kCVPixelFormatType_OneComponent16Half`,
    // which `initWithPixelBuffer:shape:` requires for the resulting array's
    // `MLMultiArrayDataTypeFloat16`; `shape`'s last dimension equals `width`
    // and the product of the remaining dimensions equals `height`, matching
    // the pixel buffer's dimensions as the API's documented contract
    // requires. Ownership: per objc2-core-ml's doc for this initializer,
    // "the pixel buffer [is] to be owned by the instance" — i.e. `inner`
    // retains `buffer` itself, so it is sound for the local `CFRetained`
    // binding to drop (release its own +1) once this function returns.
    let inner = unsafe {
      MLMultiArray::initWithPixelBuffer_shape(MLMultiArray::alloc(), &buffer, &ns_shape(shape))
    };
    Ok(Self { inner })
  }
}

#[cfg(test)]
mod tests;
