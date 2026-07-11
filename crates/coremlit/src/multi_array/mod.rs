//! Owned, typed N-dimensional CoreML arrays.

use half::f16;
use objc2::{AnyThread, ClassType, rc::Retained};
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
  // Cached at construction: an MLMultiArray's shape and strides are fixed
  // at init and never change for the array's lifetime, so one FFI read at
  // wrap time serves every later accessor call — the decoder loop reads
  // these many times per step (byte ranges, bounds checks, contiguity).
  shape: Vec<usize>,
  strides: Vec<usize>,
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
  /// Whether this OS provides `MLMultiArray`'s pixel-buffer initializer
  /// (macOS 12+), the capability behind [`Self::f16_surface`].
  pub fn supports_surface() -> bool {
    // SAFETY: `instancesRespondToSelector:` is a plain class-object query.
    unsafe {
      let responds: bool = objc2::msg_send![
        MLMultiArray::class(),
        instancesRespondToSelector: objc2::sel!(initWithPixelBuffer:shape:)
      ];
      responds
    }
  }
}

/// `shape`'s element count, rejecting `usize` overflow.
///
/// Every sizing computation in this module funnels through here (or applies
/// the same `checked_mul`/`checked_add` discipline inline for
/// stride/offset arithmetic derived from a shape) instead of
/// `Iterator::product`, which panics on overflow in debug builds and wraps
/// silently in release — either outcome would let a native FFI call proceed
/// with a size that does not match the shape it was derived from.
fn checked_element_count(shape: &[usize]) -> Result<usize, TensorError> {
  shape
    .iter()
    .try_fold(1usize, |acc, &dim| acc.checked_mul(dim))
    .ok_or_else(|| TensorError::ShapeOverflow {
      shape: shape.to_vec(),
    })
}

impl MultiArray {
  /// Allocates an array of `shape` and fills it with zero bytes.
  ///
  /// # Errors
  /// [`TensorError::UnsupportedDataType`] if `dtype` has no known element
  /// size (e.g. an unrecognized [`DataType::Unknown`] code); CoreML's
  /// `initWithShape_dataType_error` does not reject such codes itself, and
  /// an unsized dtype would make the zero-fill below unsound.
  /// [`TensorError::ShapeOverflow`] if `shape`'s element count overflows
  /// `usize`, checked before any native allocation.
  /// [`TensorError::Native`] if CoreML rejects the allocation.
  pub fn zeros(shape: &[usize], dtype: DataType) -> Result<Self, TensorError> {
    if dtype.size_of().is_none() {
      return Err(TensorError::UnsupportedDataType { dtype });
    }
    checked_element_count(shape)?;
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
    let mut this = Self::from_raw(inner);
    // MLMultiArray does not guarantee zeroed memory; mirror ArgmaxCore's
    // explicit initialValue fill.
    this.fill_bytes_zero()?;
    Ok(this)
  }

  /// Builds an array of `shape` from `data` (row-major).
  ///
  /// # Errors
  /// [`TensorError::ShapeOverflow`] if `shape`'s element count overflows
  /// `usize`. [`TensorError::ShapeMismatch`] if `data.len()` differs from
  /// the shape product; [`TensorError::Native`] if allocation fails.
  pub fn from_slice<T>(shape: &[usize], data: &[T]) -> Result<Self, TensorError>
  where
    T: Element,
  {
    let expected = checked_element_count(shape)?;
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

  /// The dimension sizes (cached at construction — the layout of an
  /// `MLMultiArray` never changes after init).
  pub fn shape(&self) -> &[usize] {
    &self.shape
  }

  /// The stride, in elements, of each dimension (cached at construction
  /// — see [`Self::shape`]).
  pub fn strides(&self) -> &[usize] {
    &self.strides
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
    for (dim, stride) in shape.iter().zip(strides).rev() {
      if *stride != expected {
        return false;
      }
      // An overflowing running product cannot equal any real stride CoreML
      // reports (those describe an actually-allocated buffer, necessarily
      // `usize`-representable), so treat overflow as a mismatch rather than
      // wrapping into a value that might coincidentally compare equal.
      let Some(next) = expected.checked_mul((*dim).max(1)) else {
        return false;
      };
      expected = next;
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
        shape: self.shape().to_vec(),
        strides: self.strides().to_vec(),
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
        shape: self.shape().to_vec(),
        strides: self.strides().to_vec(),
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
  /// [`TensorError::ShapeOverflow`] if the stride-weighted offset overflows
  /// `usize` — unreachable for indices/strides that came from a real
  /// CoreML-allocated array, but checked rather than trusted since the
  /// arithmetic is this crate's own.
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
    for ((&index, &dim), &stride) in indices.iter().zip(shape).zip(strides) {
      if index >= dim {
        return Err(TensorError::IndexOutOfBounds { index, len: dim });
      }
      let term = index
        .checked_mul(stride)
        .ok_or_else(|| TensorError::ShapeOverflow {
          shape: shape.to_vec(),
        })?;
      offset = offset
        .checked_add(term)
        .ok_or_else(|| TensorError::ShapeOverflow {
          shape: shape.to_vec(),
        })?;
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
  /// untouched rather than partially filled. [`TensorError::ShapeOverflow`]
  /// if a position's stride-weighted offset overflows `usize` (see
  /// [`Self::linear_offset`]). Dtype-mismatch failures from the typed view.
  pub fn fill_last_dim<T>(&mut self, positions: &[usize], value: T) -> Result<(), TensorError>
  where
    T: Element,
  {
    // Copy the scalars out so the cached-shape borrow ends before the
    // mutating writes below.
    let (last, stride) = {
      let shape = self.shape();
      if shape.len() >= 2 && shape[..shape.len() - 1].iter().any(|&dim| dim != 1) {
        return Err(TensorError::UnsupportedShape {
          shape: shape.to_vec(),
          reason: ShapeRequirement::LeadingDimsUnit,
        });
      }
      (
        shape.last().copied().unwrap_or(0),
        self.strides().last().copied().unwrap_or(1),
      )
    };
    for &position in positions {
      if position >= last {
        return Err(TensorError::IndexOutOfBounds {
          index: position,
          len: last,
        });
      }
    }
    for &position in positions {
      let offset = position
        .checked_mul(stride)
        .ok_or_else(|| TensorError::ShapeOverflow {
          shape: self.shape().to_vec(),
        })?;
      self.write_element(offset, value)?;
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
  /// [`TensorError::ShapeOverflow`] if the padded-row gather's offset
  /// arithmetic overflows `usize` (see [`Self::linear_offset`]).
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
    let num_rows = checked_element_count(leading_dims)?;

    for row in 0..num_rows {
      // Unravel `row` into a multi-index over `leading_dims`, row-major
      // (the last leading dimension varies fastest), folding the
      // stride-weighted sum in the same pass — no per-call index buffer,
      // so the decoder's padded-output gathers allocate nothing.
      let mut remainder = row;
      let mut row_start = 0usize;
      for i in (0..leading_dims.len()).rev() {
        let index = remainder % leading_dims[i];
        remainder /= leading_dims[i];
        let term =
          index
            .checked_mul(leading_strides[i])
            .ok_or_else(|| TensorError::ShapeOverflow {
              shape: shape.to_vec(),
            })?;
        row_start = row_start
          .checked_add(term)
          .ok_or_else(|| TensorError::ShapeOverflow {
            shape: shape.to_vec(),
          })?;
      }
      let out_start = row
        .checked_mul(last_dim)
        .ok_or_else(|| TensorError::ShapeOverflow {
          shape: shape.to_vec(),
        })?;

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
          let extra = last
            .checked_mul(last_stride)
            .ok_or_else(|| TensorError::ShapeOverflow {
              shape: shape.to_vec(),
            })?;
          let offset = row_start
            .checked_add(extra)
            .ok_or_else(|| TensorError::ShapeOverflow {
              shape: shape.to_vec(),
            })?;
          // SAFETY: dtype checked above; `offset` is a valid in-bounds
          // element offset for the same reason as the contiguous-row
          // branch, one element at a time.
          #[allow(deprecated)]
          let value = unsafe { *self.inner.dataPointer().as_ptr().cast::<T>().add(offset) };
          out[out_start + last] = value;
        }
      }
    }
    Ok(())
  }

  /// Copies this array's element data into a freshly allocated array with
  /// its own, uniquely owned native buffer.
  ///
  /// Used to de-alias a [`MultiArray`] whose buffer another live handle
  /// (an input, or another output name) also references — see
  /// [`crate::Features::from_provider`].
  ///
  /// # Errors
  /// [`TensorError::UnsupportedDataType`] if this array's element type is
  /// [`DataType::Unknown`] (no [`Element`] impl exists to copy through, so
  /// there is no `T` to gather into). Propagates [`Self::copy_into`]/
  /// [`Self::from_slice`] failures otherwise.
  pub(crate) fn deep_copy(&self) -> Result<Self, TensorError> {
    let shape = self.shape();
    fn copy_typed<T: Element + Default>(
      this: &MultiArray,
      shape: &[usize],
    ) -> Result<MultiArray, TensorError> {
      let mut buf = vec![T::default(); this.count()];
      this.copy_into(&mut buf)?;
      MultiArray::from_slice(shape, &buf)
    }
    match self.data_type() {
      DataType::F16 => copy_typed::<f16>(self, shape),
      DataType::F32 => copy_typed::<f32>(self, shape),
      DataType::F64 => copy_typed::<f64>(self, shape),
      DataType::I32 => copy_typed::<i32>(self, shape),
      dtype @ DataType::Unknown(_) => Err(TensorError::UnsupportedDataType { dtype }),
    }
  }

  /// This array's addressed byte region, as opaque `[start, end)` values.
  ///
  /// Never dereferenced through this crate; used only to detect when two
  /// [`MultiArray`]s alias OVERLAPPING storage (see
  /// [`crate::Features::from_provider`]) — exact-pointer equality alone
  /// would miss a view offset inside another array's buffer. The extent is
  /// `1 + Σ((dim − 1) · stride)` elements — the highest linear offset any
  /// index tuple can reach, valid for ARBITRARY stride orderings (a
  /// row-major-only `shape[0] · strides[0]` would under-cover e.g. strides
  /// `[1, 100]`, where the farthest element lies along the LAST axis).
  /// Trust-boundary arithmetic saturates, so a provider reporting absurd
  /// geometry yields a conservatively HUGE range, which can only force an
  /// unnecessary copy — never miss a real overlap.
  pub(crate) fn byte_range(&self) -> (usize, usize) {
    // SAFETY: `dataPointer` is a plain accessor message send on a live
    // object; the returned pointer is used only as an opaque address
    // value here, never read through.
    #[allow(deprecated)]
    let start = unsafe { self.inner.dataPointer().as_ptr() as usize };
    let elem = self.data_type().size_of().unwrap_or(1);
    let span_elements = self
      .shape()
      .iter()
      .zip(self.strides())
      .fold(1usize, |acc, (&dim, &stride)| {
        acc.saturating_add(dim.saturating_sub(1).saturating_mul(stride))
      })
      .max(self.count());
    let span_bytes = span_elements.saturating_mul(elem).max(1);
    (start, start.saturating_add(span_bytes))
  }

  pub(crate) fn raw(&self) -> &MLMultiArray {
    &self.inner
  }

  // INVARIANT: callers must pass the sole `Retained` reference to this
  // array. `Send` and `as_slice_mut`'s exclusivity both assume no aliased
  // handle exists (Retained is Clone, so this cannot be enforced by type).
  pub(crate) fn from_raw(inner: Retained<MLMultiArray>) -> Self {
    // SAFETY: accessor message sends on a live object; both values are
    // fixed at the array's init, so reading them once here is exhaustive.
    let shape = unsafe { inner.shape() }
      .iter()
      .map(|n| n.as_usize())
      .collect();
    // SAFETY: as above — accessor message send on the same live object.
    let strides = unsafe { inner.strides() }
      .iter()
      .map(|n| n.as_usize())
      .collect();
    Self {
      inner,
      shape,
      strides,
    }
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

  /// # Errors
  /// [`TensorError::ShapeOverflow`] if `count() * size_of(data_type())`
  /// overflows `usize` — the element count itself was already checked
  /// against the shape in `zeros`, but multiplying by byte size is a
  /// distinct overflow surface (e.g. a large [`DataType::F64`] count).
  fn fill_bytes_zero(&mut self) -> Result<(), TensorError> {
    let byte_len = self
      .count()
      .checked_mul(
        self
          .data_type()
          .size_of()
          .expect("constructors validate the data type"),
      )
      .ok_or_else(|| TensorError::ShapeOverflow {
        shape: self.shape().to_vec(),
      })?;
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
    Ok(())
  }
}

/// IOSurface-backed construction (ANE-efficient f16 I/O).
impl MultiArray {
  /// Allocates an IOSurface-backed half-precision array.
  ///
  /// CoreML shares IOSurface-backed `f16` buffers with the ANE without
  /// copies; WhisperKit allocates every f16 tensor this way. Width is the
  /// last dimension exactly; height is the product of the leading
  /// dimensions (every dimension must be nonzero).
  ///
  /// The IOSurface may pad each row out to a platform-chosen alignment, so
  /// the result is not guaranteed [`Self::is_contiguous`]. When padded,
  /// bulk access via [`Self::as_slice`]/[`Self::as_slice_mut`] returns
  /// [`TensorError::NonContiguous`] rather than silently reading/writing
  /// through the padding; element access ([`Self::fill_at`],
  /// [`Self::fill_last_dim`]) and CoreML's own prediction APIs are
  /// stride-aware and read/write the correct bytes either way. As with
  /// [`Self::zeros`], this constructor zero-fills the buffer — including any
  /// inter-row padding — before returning, so every logical element is safe
  /// to read immediately, whether or not the layout turned out padded.
  ///
  /// # Errors
  /// [`TensorError::UnsupportedShape`] (reason
  /// [`ShapeRequirement::NonEmpty`]) if `shape` is empty — a rank-0 shape
  /// has no last dimension to serve as the pixel buffer width, so
  /// [`Self::shape`]/[`Self::linear_offset`] on the result would be
  /// nonsensical rather than merely unusual.
  /// [`TensorError::UnsupportedShape`] (reason
  /// [`ShapeRequirement::NonZeroDims`]) if any dimension is zero — the
  /// initializer requires the pixel width to equal the final dimension
  /// exactly, and a zero-element surface has no valid width/height pair.
  /// [`TensorError::ShapeOverflow`] if `shape`'s element count overflows
  /// `usize`, checked (and `height` derived from the checked leading
  /// product) before any native allocation.
  /// [`TensorError::PixelBuffer`] if CoreVideo rejects the pixel buffer
  /// allocation or CPU-access locking.
  /// [`TensorError::SurfaceUnsupported`] before macOS 12, where
  /// `MLMultiArray` has no pixel-buffer initializer — probed at runtime so
  /// the unrecognized selector surfaces as an error, never an Objective-C
  /// exception.
  pub fn f16_surface(shape: &[usize]) -> Result<Self, TensorError> {
    use objc2_core_foundation::{CFDictionary, CFRetained, CFType};
    use objc2_core_video::{
      CVPixelBuffer, CVPixelBufferCreate, kCVPixelBufferIOSurfacePropertiesKey,
      kCVPixelFormatType_OneComponent16Half, kCVReturnSuccess,
    };

    if !Self::supports_surface() {
      return Err(TensorError::SurfaceUnsupported);
    }

    if shape.is_empty() {
      return Err(TensorError::UnsupportedShape {
        shape: Vec::new(),
        reason: ShapeRequirement::NonEmpty,
      });
    }
    // `initWithPixelBuffer:shape:` requires the pixel width to EQUAL the
    // final shape dimension — a zero dimension would force a clamped width
    // that violates that contract (an Objective-C exception from safe
    // code), and a zero-element surface is meaningless anyway.
    if shape.contains(&0) {
      return Err(TensorError::UnsupportedShape {
        shape: shape.to_vec(),
        reason: ShapeRequirement::NonZeroDims,
      });
    }

    // `initWithPixelBuffer:shape:` requires the product of every dimension
    // before the last to equal the pixel buffer's height; derive `height`
    // directly from that checked product (rather than a checked total
    // divided by `width`), then separately confirm the two multiply back
    // together without overflow. `shape` is non-empty and all-nonzero
    // (checked above), so `shape.len() - 1` cannot underflow and `width`
    // is the exact final dimension with no clamping.
    let width = *shape.last().expect("shape checked non-empty above");
    let height = checked_element_count(&shape[..shape.len() - 1])?;
    height
      .checked_mul(width)
      .ok_or_else(|| TensorError::ShapeOverflow {
        shape: shape.to_vec(),
      })?;

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
    // Zero the pixel buffer's ENTIRE allocation before CoreML wraps it —
    // `CVPixelBufferGetDataSize` is the allocator's own byte count, so this
    // covers row-tail padding on every rank (a rank-1 array's sole stride
    // is 1 and would under-cover the padded row; deriving the span from
    // MLMultiArray strides cannot see allocator padding past the logical
    // extent). CoreML/CoreVideo do not guarantee zeroed memory, and
    // `as_slice`/`read_at`/`copy_into` on the result must never observe
    // uninitialized bytes.
    // SAFETY: `buffer` is the live pixel buffer created above; lock/base/
    // size/unlock is the documented CPU-access protocol, base is non-null
    // after a successful lock of a successfully created buffer, and
    // `write_bytes` stays within `CVPixelBufferGetDataSize` bytes of the
    // locked base. All-zero bits are valid `f16` (0.0).
    unsafe {
      use objc2_core_video::{
        CVPixelBufferGetBaseAddress, CVPixelBufferGetDataSize, CVPixelBufferLockBaseAddress,
        CVPixelBufferLockFlags, CVPixelBufferUnlockBaseAddress,
      };
      let lock = CVPixelBufferLockBaseAddress(&buffer, CVPixelBufferLockFlags::empty());
      if lock != kCVReturnSuccess {
        return Err(TensorError::PixelBuffer { code: lock });
      }
      let base = CVPixelBufferGetBaseAddress(&buffer);
      if base.is_null() {
        CVPixelBufferUnlockBaseAddress(&buffer, CVPixelBufferLockFlags::empty());
        return Err(TensorError::PixelBuffer { code: lock });
      }
      core::ptr::write_bytes(base.cast::<u8>(), 0, CVPixelBufferGetDataSize(&buffer));
      CVPixelBufferUnlockBaseAddress(&buffer, CVPixelBufferLockFlags::empty());
    }

    // SAFETY: `buffer`'s pixel format matches Float16, `shape`'s last
    // dimension equals `width` exactly (zero dims were rejected above, so
    // no `.max(1)` clamp can diverge from the real dimension), and the
    // leading product equals `height` — the initializer's documented
    // contract. Ownership: the instance retains the buffer, so the local
    // `CFRetained` may drop after this call.
    let inner = unsafe {
      MLMultiArray::initWithPixelBuffer_shape(MLMultiArray::alloc(), &buffer, &ns_shape(shape))
    };
    Ok(Self::from_raw(inner))
  }
}

#[cfg(test)]
mod tests;
