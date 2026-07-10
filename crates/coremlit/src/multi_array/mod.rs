//! Owned, typed N-dimensional CoreML arrays.

use half::f16;
use objc2::{AnyThread, rc::Retained};
use objc2_core_ml::{MLMultiArray, MLMultiArrayDataType};
use objc2_foundation::{NSArray, NSNumber};

use crate::{DataType, NsErrorInfo, TensorError};

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
  /// [`TensorError::Native`] if CoreML rejects the allocation.
  pub fn zeros(shape: &[usize], dtype: DataType) -> Result<Self, TensorError> {
    // SAFETY: valid shape array and data-type code; result checked.
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
  /// [`Self::data_type`].
  pub fn as_slice<T>(&self) -> Result<&[T], TensorError>
  where
    T: Element,
  {
    self.check_dtype::<T>()?;
    // SAFETY: dtype checked; CPU-backed buffer is contiguous for
    // default-stride arrays; lifetime tied to &self. `dataPointer` is the
    // only stable way to hand out a borrow (the block-based accessors
    // scope the pointer to a closure); Swift WhisperKit does the same.
    #[allow(deprecated)]
    Ok(unsafe {
      core::slice::from_raw_parts(self.inner.dataPointer().as_ptr().cast(), self.count())
    })
  }

  /// Borrows the elements as `&mut [T]`.
  ///
  /// # Errors
  /// [`TensorError::DataTypeMismatch`] if `T` differs from
  /// [`Self::data_type`].
  pub fn as_slice_mut<T>(&mut self) -> Result<&mut [T], TensorError>
  where
    T: Element,
  {
    self.check_dtype::<T>()?;
    // SAFETY: as in `as_slice`, plus exclusivity via &mut self.
    #[allow(deprecated)]
    Ok(unsafe {
      core::slice::from_raw_parts_mut(self.inner.dataPointer().as_ptr().cast(), self.count())
    })
  }

  #[allow(dead_code)] // consumed from Task 7 (Features)
  pub(crate) fn raw(&self) -> &MLMultiArray {
    &self.inner
  }

  #[allow(dead_code)] // consumed from Task 7 (Features)
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

  fn fill_bytes_zero(&mut self) {
    let byte_len = self.count() * self.data_type().size_of().unwrap_or(0);
    // SAFETY: writing zero bytes within the allocation's length.
    #[allow(deprecated)]
    unsafe {
      core::ptr::write_bytes(self.inner.dataPointer().as_ptr().cast::<u8>(), 0, byte_len);
    }
  }
}

#[cfg(test)]
mod tests;
