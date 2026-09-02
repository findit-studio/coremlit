use half::f16;

use super::*;
use crate::{
  AllocationFailed, DataType, DataTypeMismatch, IndexOutOfBounds, RankMismatch, ShapeMismatch,
  ShapeRequirement, TensorError, UnsupportedShape,
};

#[test]
fn zeros_has_shape_count_dtype_and_zero_content() {
  let arr = MultiArray::zeros(&[2, 3, 4], DataType::F32).unwrap();
  assert_eq!(arr.shape(), vec![2, 3, 4]);
  assert_eq!(arr.count(), 24);
  assert_eq!(arr.data_type(), DataType::F32);
  assert!(arr.as_slice::<f32>().unwrap().iter().all(|v| *v == 0.0));
}

#[test]
fn from_slice_round_trips_f32() {
  let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
  let arr = MultiArray::from_slice(&[2, 3], &data).unwrap();
  assert_eq!(arr.as_slice::<f32>().unwrap(), &data);
}

#[test]
fn from_slice_round_trips_f16_and_i32() {
  let h = [f16::from_f32(0.5), f16::from_f32(-1.0)];
  let arr = MultiArray::from_slice(&[2], &h).unwrap();
  assert_eq!(arr.as_slice::<f16>().unwrap(), &h);
  assert_eq!(arr.data_type(), DataType::F16);

  let ints = [7i32, -7];
  let arr = MultiArray::from_slice(&[1, 2], &ints).unwrap();
  assert_eq!(arr.as_slice::<i32>().unwrap(), &ints);
}

#[test]
fn wrong_view_type_is_dtype_mismatch() {
  let arr = MultiArray::zeros(&[4], DataType::F32).unwrap();
  let err = arr.as_slice::<i32>().unwrap_err();
  assert_eq!(
    err,
    TensorError::DataTypeMismatch(DataTypeMismatch::new(DataType::I32, DataType::F32))
  );
}

#[test]
fn from_slice_rejects_shape_element_mismatch() {
  let err = MultiArray::from_slice(&[2, 2], &[1.0f32]).unwrap_err();
  assert_eq!(err, TensorError::ShapeMismatch(ShapeMismatch::new(4, 1)));
}

#[test]
fn as_slice_mut_writes_are_visible() {
  let mut arr = MultiArray::zeros(&[3], DataType::F32).unwrap();
  arr.as_slice_mut::<f32>().unwrap()[1] = 9.5;
  assert_eq!(arr.as_slice::<f32>().unwrap()[1], 9.5);
}

#[test]
fn zeros_rejects_unknown_dtype() {
  let err = MultiArray::zeros(&[4], DataType::Unknown(0)).unwrap_err();
  assert_eq!(err, TensorError::UnsupportedDataType(DataType::Unknown(0)));
}

#[test]
fn linear_offset_uses_strides() {
  let arr = MultiArray::zeros(&[2, 3, 4], DataType::F32).unwrap();
  assert_eq!(arr.linear_offset(&[0, 0, 0]).unwrap(), 0);
  assert_eq!(arr.linear_offset(&[1, 2, 3]).unwrap(), 23);
  assert_eq!(
    arr.linear_offset(&[1, 2]).unwrap_err(),
    TensorError::RankMismatch(RankMismatch::new(3, 2))
  );
  assert_eq!(
    arr.linear_offset(&[0, 3, 0]).unwrap_err(),
    TensorError::IndexOutOfBounds(IndexOutOfBounds::new(3, 3))
  );
}

#[test]
fn fill_at_writes_one_element() {
  let mut arr = MultiArray::zeros(&[2, 2], DataType::F32).unwrap();
  arr.fill_at(&[1, 0], 7.0f32).unwrap();
  assert_eq!(arr.as_slice::<f32>().unwrap(), &[0.0, 0.0, 7.0, 0.0]);
}

#[test]
fn fill_last_dim_writes_positions() {
  let mut arr = MultiArray::zeros(&[1, 1, 4], DataType::F32).unwrap();
  arr.fill_last_dim(&[0, 2], 1.5f32).unwrap();
  assert_eq!(arr.as_slice::<f32>().unwrap(), &[1.5, 0.0, 1.5, 0.0]);
}

#[test]
fn fill_last_dim_rejects_non_unit_leading_dims() {
  let mut arr = MultiArray::zeros(&[2, 3, 4], DataType::F32).unwrap();
  let err = arr.fill_last_dim(&[0], 1.0f32).unwrap_err();
  assert_eq!(
    err,
    TensorError::UnsupportedShape(UnsupportedShape::new(
      vec![2, 3, 4],
      ShapeRequirement::LeadingDimsUnit
    ))
  );
}

#[test]
fn fill_last_dim_oob_position_leaves_array_untouched() {
  let mut arr = MultiArray::zeros(&[1, 1, 4], DataType::F32).unwrap();
  let err = arr.fill_last_dim(&[0, 2, 10], 1.5f32).unwrap_err();
  assert_eq!(
    err,
    TensorError::IndexOutOfBounds(IndexOutOfBounds::new(10, 4))
  );
  assert!(arr.as_slice::<f32>().unwrap().iter().all(|v| *v == 0.0));
}

#[test]
fn f16_surface_is_f16_and_writable() {
  let mut arr = MultiArray::f16_surface(&[1, 2, 1, 4]).unwrap();
  assert_eq!(arr.data_type(), DataType::F16);
  assert_eq!(arr.shape(), vec![1, 2, 1, 4]);
  let half_one = f16::from_f32(1.0);
  if arr.is_contiguous() {
    arr.as_slice_mut::<f16>().unwrap().fill(half_one);
    assert!(
      arr
        .as_slice::<f16>()
        .unwrap()
        .iter()
        .all(|v| *v == half_one)
    );
    return;
  }
  // Row-padded on this host: bulk slice views are rejected (see
  // `padded_surface_rejects_flat_views_but_fills_elementwise`). Write every
  // element individually and cross-check through CoreML's own
  // `objectAtIndexedSubscript`, which — per objc2-core-ml's doc ("Get a
  // value by its linear index (assumes C-style index ordering)") — takes a
  // logical, shape-derived C-order position and internally re-applies the
  // array's real strides, an independent read path from this crate's own
  // `linear_offset`/`write_element`.
  let shape = arr.shape().to_vec();
  for i0 in 0..shape[0] {
    for i1 in 0..shape[1] {
      for i2 in 0..shape[2] {
        for i3 in 0..shape[3] {
          arr.fill_at(&[i0, i1, i2, i3], half_one).unwrap();
        }
      }
    }
  }
  for linear in 0..arr.count() {
    // SAFETY: accessor send on a live object; `linear` is in `0..count()`,
    // matching `objectAtIndexedSubscript`'s documented C-style linear
    // indexing contract.
    let value = unsafe { arr.raw().objectAtIndexedSubscript(linear as isize) };
    assert_eq!(value.floatValue(), half_one.to_f32());
  }
}

#[test]
fn f16_surface_rejects_empty_shape() {
  let err = MultiArray::f16_surface(&[]).unwrap_err();
  assert_eq!(
    err,
    TensorError::UnsupportedShape(UnsupportedShape::new(
      Vec::new(),
      ShapeRequirement::NonEmpty
    ))
  );
}

#[test]
fn f16_surface_reports_pixel_buffer_backing() {
  let arr = MultiArray::f16_surface(&[1, 2, 1, 4]).unwrap();
  // SAFETY: accessor send on a live object.
  assert!(unsafe { arr.raw().pixelBuffer() }.is_some());
}

#[test]
fn copy_into_and_read_at_round_trip_contiguous() {
  let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
  let arr = MultiArray::from_slice(&[2, 3], &data).unwrap();
  let mut out = [0.0f32; 6];
  arr.copy_into(&mut out).unwrap();
  assert_eq!(out, data);
  assert_eq!(arr.read_at::<f32>(&[0, 0]).unwrap(), 1.0);
  assert_eq!(arr.read_at::<f32>(&[1, 2]).unwrap(), 6.0);
}

#[test]
fn copy_into_and_read_at_expose_padded_second_row() {
  let mut arr = MultiArray::f16_surface(&[1, 2, 1, 4]).unwrap();
  if arr.is_contiguous() {
    // Row padding is an allocator decision; nothing to assert on this host.
    return;
  }
  let row0_value = f16::from_f32(-1.25);
  let row1_value = f16::from_f32(2.5);
  arr.fill_at(&[0, 0, 0, 0], row0_value).unwrap();
  arr.fill_at(&[0, 1, 0, 3], row1_value).unwrap();

  assert_eq!(arr.read_at::<f16>(&[0, 0, 0, 0]).unwrap(), row0_value);
  assert_eq!(arr.read_at::<f16>(&[0, 1, 0, 3]).unwrap(), row1_value);

  let mut out = [f16::from_f32(0.0); 8];
  arr.copy_into(&mut out).unwrap();
  // Row-major flatten of [1,2,1,4]: [0,0,0,0] -> 0, [0,1,0,3] -> 7. Index 7
  // falls inside row 1 (i1 == 1), the padded row `as_slice` refuses to
  // expose — the exact gap `copy_into`/`read_at` exist to close.
  assert_eq!(out[0], row0_value);
  assert_eq!(out[7], row1_value);
}

#[test]
fn zeros_arrays_are_contiguous_and_sliceable() {
  let arr = MultiArray::zeros(&[2, 3], DataType::F32).unwrap();
  assert!(arr.is_contiguous());
  assert!(arr.as_slice::<f32>().is_ok());
}

#[test]
fn padded_surface_rejects_flat_views_but_fills_elementwise() {
  let mut arr = MultiArray::f16_surface(&[1, 2, 1, 4]).unwrap();
  if arr.is_contiguous() {
    // Row padding is an allocator decision; nothing to assert on this host.
    return;
  }
  assert!(matches!(
    arr.as_slice::<f16>(),
    Err(TensorError::NonContiguous(_))
  ));
  assert!(matches!(
    arr.as_slice_mut::<f16>(),
    Err(TensorError::NonContiguous(_))
  ));
  arr.fill_at(&[0, 1, 0, 3], f16::from_f32(2.5)).unwrap();
  let offset = arr.linear_offset(&[0, 1, 0, 3]).unwrap();
  // `fill_at` wrote through this same stride-derived offset, so this only
  // confirms the arithmetic is self-consistent; the independent read-back
  // through CoreML's own `objectAtIndexedSubscript` lives in
  // `f16_surface_is_f16_and_writable`.
  assert_eq!(arr.strides()[1] + 3, offset);
}

#[test]
fn fill_last_dim_accepts_rank_one() {
  let mut arr = MultiArray::zeros(&[4], DataType::F32).unwrap();
  arr.fill_last_dim(&[0, 2], 1.5f32).unwrap();
  assert_eq!(arr.as_slice::<f32>().unwrap(), &[1.5, 0.0, 1.5, 0.0]);
}

#[test]
fn unsupported_shape_displays_reason() {
  let err = MultiArray::zeros(&[2, 3], DataType::F32)
    .unwrap()
    .fill_last_dim(&[0], 1.0f32)
    .unwrap_err();
  assert_eq!(
    err.to_string(),
    "shape [2, 3] is unsupported: all dimensions before the last must be 1"
  );
}

#[test]
fn f16_surface_padded_elements_are_zero_before_any_write() {
  // Same reachable history as the review's PoC: read immediately after
  // construction, before any write, through the stride-aware `read_at` path
  // that bypasses the contiguity guard.
  let arr = MultiArray::f16_surface(&[1, 4]).unwrap();
  if arr.is_contiguous() {
    // Row padding is an allocator decision; nothing to assert on this host
    // (see `f16_surface_contiguous_is_zero_before_any_write` below).
    return;
  }
  let shape = arr.shape();
  for i0 in 0..shape[0] {
    for i1 in 0..shape[1] {
      assert_eq!(
        arr.read_at::<f16>(&[i0, i1]).unwrap(),
        f16::from_f32(0.0),
        "logical index [{i0}, {i1}] was not zeroed"
      );
    }
  }
}

#[test]
fn f16_surface_contiguous_is_zero_before_any_write() {
  // A wide-enough row already satisfies the platform's IOSurface row
  // alignment with no padding, so this exercises the bulk `as_slice` path
  // (only reachable when contiguous) reading immediately after
  // construction, before any write.
  let arr = MultiArray::f16_surface(&[1, 64]).unwrap();
  if !arr.is_contiguous() {
    // Row alignment padded this host's buffer; the padded branch above
    // already covers the uninitialized-read invariant via `read_at`.
    return;
  }
  assert!(
    arr
      .as_slice::<f16>()
      .unwrap()
      .iter()
      .all(|v| *v == f16::from_f32(0.0))
  );
}

#[test]
fn f16_surface_zero_fills_every_logical_element_at_every_shape() {
  // `f16_surface`'s contract at the shapes the whisper alignment gather uses,
  // including the tall 225-row accumulator: every logical element reads zero
  // immediately after construction, whether or not CoreVideo padded the rows.
  //
  // This asserts what THIS crate wrote, and nothing about what CoreVideo left
  // behind first — the constructor is write-only by design (reading a fresh
  // pixel buffer's bytes is UB, whatever the read is spelled as), so there is
  // no prior content any test could legitimately observe.
  for shape in [[1usize, 4], [3, 9], [225, 100], [224, 100]] {
    let arr = MultiArray::f16_surface(&shape).unwrap();
    assert_eq!(arr.shape(), shape);
    for i0 in 0..shape[0] {
      for i1 in 0..shape[1] {
        assert_eq!(
          arr.read_at::<f16>(&[i0, i1]).unwrap(),
          f16::from_f32(0.0),
          "shape {shape:?} index [{i0}, {i1}] was not zeroed"
        );
      }
    }
  }
}

#[test]
fn zeros_rejects_shape_overflow() {
  let err = MultiArray::zeros(&[usize::MAX, 2], DataType::F32).unwrap_err();
  assert_eq!(err, TensorError::ShapeOverflow(vec![usize::MAX, 2]));
}

#[test]
fn from_slice_rejects_shape_overflow() {
  let data = [1.0f32];
  let err = MultiArray::from_slice(&[usize::MAX, 2], &data).unwrap_err();
  assert_eq!(err, TensorError::ShapeOverflow(vec![usize::MAX, 2]));
}

#[test]
fn f16_surface_rejects_shape_overflow() {
  let huge = usize::MAX / 4 + 2;
  let err = MultiArray::f16_surface(&[huge, 4]).unwrap_err();
  assert_eq!(err, TensorError::ShapeOverflow(vec![huge, 4]));
}

#[test]
fn surface_probe_is_true_on_this_host() {
  assert!(MultiArray::supports_surface());
}

#[test]
fn f16_surface_rejects_zero_dimensions() {
  for shape in [&[0usize][..], &[1, 0], &[0, 4]] {
    let err = MultiArray::f16_surface(shape).unwrap_err();
    assert_eq!(
      err,
      TensorError::UnsupportedShape(UnsupportedShape::new(
        shape.to_vec(),
        ShapeRequirement::NonZeroDims
      )),
      "shape {shape:?}"
    );
  }
}

#[test]
fn byte_range_covers_non_major_strides() {
  // Strides [1, 100] put the farthest element of a [2, 2] array at linear
  // offset 101 — a row-major `dim0 * stride0` extent (= 2) would miss it.
  use objc2::AnyThread;
  use objc2_core_ml::{MLMultiArray, MLMultiArrayDataType};
  use objc2_foundation::{NSArray, NSNumber};
  let dims: Vec<_> = [2usize, 2]
    .iter()
    .map(|d| NSNumber::new_usize(*d))
    .collect();
  let strides: Vec<_> = [1usize, 100]
    .iter()
    .map(|d| NSNumber::new_usize(*d))
    .collect();
  // SAFETY: valid shape/stride arrays; the initializer allocates backing
  // storage sized for the strides it is given.
  let raw = unsafe {
    MLMultiArray::initWithShape_dataType_strides(
      MLMultiArray::alloc(),
      &NSArray::from_retained_slice(&dims),
      MLMultiArrayDataType(DataType::F32.to_raw()),
      &NSArray::from_retained_slice(&strides),
    )
  };
  let arr = MultiArray::from_raw(raw);
  let (start, end) = arr.byte_range();
  // 1 + (2-1)*1 + (2-1)*100 = 102 elements minimum.
  assert!(end - start >= 102 * 4, "extent {} too small", end - start);
}

/// **FALSIFIER (red first).** [`MultiArray::deep_copy`] gathers through a Rust
/// buffer sized by [`MultiArray::count`], and for an array extracted from a
/// prediction that count is the product of the axes the ARTIFACT declared —
/// checked to fit `usize` by `checked_element_count`, and bounded by nothing
/// else. `vec![T::default(); n]` answered a count the allocator will not serve
/// by ABORTING the process, which no `expect_err` can observe and no caller of
/// `Result<_, TensorError>` can handle.
///
/// The regime cannot be reached through a real array — allocating one that big
/// is exactly what fails — so the helper is driven directly at two absurd
/// lengths, one on each side of `Vec`'s own capacity arithmetic. The assertion
/// is that an `Err` comes back at all.
#[test]
fn the_gather_buffer_refuses_rather_than_aborting() {
  // `usize::MAX` f32s is `usize::MAX * 4` bytes: the length cannot even be
  // turned into a layout, and `Vec` reports `CapacityOverflow`.
  let error = gather_buffer::<f32>(usize::MAX)
    .expect_err("a length whose byte size leaves `usize` has no buffer");
  assert_eq!(
    error,
    TensorError::AllocationFailed(AllocationFailed::new(usize::MAX, DataType::F32))
  );

  // `usize::MAX / 8` f32s is a layout `Vec` will happily describe — under
  // `isize::MAX` bytes — and that no allocator will satisfy, so this is the
  // arm that reaches a real `AllocError`.
  let beyond_memory = usize::MAX / 8;
  let error = gather_buffer::<f16>(beyond_memory)
    .expect_err("a buffer the allocator refuses is an error, not an abort");
  assert_eq!(
    error,
    TensorError::AllocationFailed(AllocationFailed::new(beyond_memory, DataType::F16))
  );
  assert!(
    error.to_string().contains(&beyond_memory.to_string()),
    "the refusal must name the length that was asked for, got {error}"
  );

  // A length that CAN be met still comes back zeroed and exactly that long, so
  // the fallible path did not change what the buffer is.
  let buf = gather_buffer::<f32>(6).expect("six f32s fit");
  assert_eq!(buf.len(), 6);
  assert!(buf.iter().all(|v| v.to_bits() == 0.0f32.to_bits()));
}

/// The caller the helper above exists for: `deep_copy` still reproduces the
/// content into a buffer of its own, so rewiring it through a fallible
/// reservation did not change what a de-aliased array is.
#[test]
fn deep_copy_reproduces_content_in_its_own_buffer() {
  let data = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
  let arr = MultiArray::from_slice(&[2, 3], &data).unwrap();
  let copy = arr.deep_copy().unwrap();
  assert_eq!(copy.shape(), &[2, 3]);
  assert_eq!(copy.data_type(), DataType::F32);
  assert_eq!(copy.as_slice::<f32>().unwrap(), &data);
  // Both are live here, so distinct addresses prove the copy owns its storage
  // rather than sharing the original's.
  assert_ne!(
    arr.byte_range().0,
    copy.byte_range().0,
    "a deep copy must not share the original's buffer"
  );
}
