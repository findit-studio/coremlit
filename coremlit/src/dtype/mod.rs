//! Element types of CoreML multi-arrays.

/// Element type of a [`MultiArray`](crate::MultiArray).
///
/// Coded vocabulary over `MLMultiArrayDataType` raw values; unknown codes
/// round-trip losslessly through [`Self::Unknown`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum DataType {
  /// IEEE 754 half precision.
  F16,
  /// IEEE 754 single precision.
  F32,
  /// IEEE 754 double precision.
  F64,
  /// 32-bit signed integer.
  I32,
  /// A raw `MLMultiArrayDataType` value this crate does not model.
  Unknown(isize),
}

const RAW_F64: isize = 0x10000 | 64;
const RAW_F32: isize = 0x10000 | 32;
const RAW_F16: isize = 0x10000 | 16;
const RAW_I32: isize = 0x20000 | 32;

impl DataType {
  /// Stable name; `Unknown(_)` renders as `"unknown"`.
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::F16 => "float16",
      Self::F32 => "float32",
      Self::F64 => "float64",
      Self::I32 => "int32",
      Self::Unknown(_) => "unknown",
    }
  }

  /// Size of one element in bytes; `None` for [`Self::Unknown`].
  #[inline(always)]
  pub const fn size_of(&self) -> Option<usize> {
    match self {
      Self::F16 => Some(2),
      Self::F32 | Self::I32 => Some(4),
      Self::F64 => Some(8),
      Self::Unknown(_) => None,
    }
  }

  /// The raw `MLMultiArrayDataType` value.
  #[inline(always)]
  pub const fn to_raw(self) -> isize {
    match self {
      Self::F64 => RAW_F64,
      Self::F32 => RAW_F32,
      Self::F16 => RAW_F16,
      Self::I32 => RAW_I32,
      Self::Unknown(raw) => raw,
    }
  }

  /// Lossless inverse of [`Self::to_raw`].
  #[inline(always)]
  pub const fn from_raw(raw: isize) -> Self {
    match raw {
      RAW_F64 => Self::F64,
      RAW_F32 => Self::F32,
      RAW_F16 => Self::F16,
      RAW_I32 => Self::I32,
      other => Self::Unknown(other),
    }
  }
}

#[cfg(test)]
mod tests;
