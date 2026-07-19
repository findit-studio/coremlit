//! The 384-dim L2-normalized granite [`Embedding`].
//!
//! The type and its numeric contract (f64-accumulated normalization,
//! `is_close` / `is_close_cosine`, no `PartialEq`) mirror `clap`'s `Embedding`
//! deliberately, so the two embedding surfaces read identically; only the
//! dimension (384) and the names change. (Plain-text reference — granite builds
//! without the `clap` feature, so its docs must not link across it.)

use core::{fmt, ops::Deref};

use crate::embeddings::granite::error::{Error, Result};

/// Dimensionality of a granite text embedding (the ModernBERT encoder projects
/// its CLS token to 384). Pinned from the converted graph's output contract
/// (`tests/granite/model_io.rs`).
pub const EMBEDDING_DIM: usize = 384;

/// Norm-tolerance budget for the trusted-path unit-norm check
/// ([`Embedding::try_from_unit_slice`]). Worst case `384 · ulp(1) ≈ 4.6e-5`,
/// rounded to `1e-4` — the same budget the CLAP embedding uses, comfortably
/// above the 384-term accumulation error.
pub(crate) const NORM_BUDGET: f32 = 1e-4;

/// A 384-dim L2-normalized granite embedding.
///
/// Returned by [`crate::embeddings::granite::TextEmbedder::embed`]. The
/// unit-norm invariant holds within fp32 ULP.
///
/// # Compile-fail contracts
///
/// `Embedding` exposes no `DIM` associated const (use the module-level
/// [`EMBEDDING_DIM`]):
///
/// ```compile_fail
/// let _ = coremlit::embeddings::granite::Embedding::DIM;
/// ```
///
/// `Embedding` does not implement `PartialEq` — f32 outputs of an ML model are
/// not bit-stable across runs / threads / OSes; use [`Embedding::is_close`] or
/// [`Embedding::is_close_cosine`]:
///
/// ```compile_fail
/// # let mut s = [0.0_f32; 384]; s[0] = 1.0;
/// # let a = coremlit::embeddings::granite::Embedding::from_slice_normalizing(&s).unwrap();
/// # let b = a.clone();
/// let _ = a == b;
/// ```
#[derive(Clone)]
#[repr(transparent)]
pub struct Embedding {
  inner: [f32; EMBEDDING_DIM],
}

impl Embedding {
  /// Length of the embedding (384).
  #[inline]
  pub const fn dim(&self) -> usize {
    self.inner.len()
  }

  /// Borrow the embedding as a slice.
  #[inline]
  pub const fn as_slice(&self) -> &[f32] {
    self.inner.as_slice()
  }

  /// Owned conversion to a `Vec<f32>`. Allocates.
  #[inline]
  pub fn to_vec(&self) -> Vec<f32> {
    self.inner.to_vec()
  }

  /// Reconstruct from a stored unit vector. Validates length, finiteness, AND
  /// unit-norm (`(norm² − 1).abs() ≤ ``NORM_BUDGET`).
  ///
  /// # Errors
  /// [`Error::EmbeddingDimMismatch`] if `s.len() != `[`EMBEDDING_DIM`];
  /// [`Error::NonFiniteEmbedding`] on any non-finite component;
  /// [`Error::EmbeddingNotUnitNorm`] if the norm is outside the budget.
  pub fn try_from_unit_slice(s: &[f32]) -> Result<Self> {
    if s.len() != EMBEDDING_DIM {
      return Err(Error::EmbeddingDimMismatch {
        expected: EMBEDDING_DIM,
        got: s.len(),
      });
    }
    for (i, &v) in s.iter().enumerate() {
      if !v.is_finite() {
        return Err(Error::NonFiniteEmbedding { component_index: i });
      }
    }
    let norm_sq: f32 = s.iter().map(|x| x * x).sum();
    let dev = (norm_sq - 1.0).abs();
    if dev > NORM_BUDGET {
      return Err(Error::EmbeddingNotUnitNorm {
        norm_sq_deviation: dev,
      });
    }
    let mut inner = [0.0f32; EMBEDDING_DIM];
    inner.copy_from_slice(s);
    Ok(Self { inner })
  }

  /// Construct from any non-zero finite slice, re-normalizing to unit length.
  ///
  /// The norm is accumulated in f64 so any finite f32 input normalizes without
  /// intermediate overflow (e.g. `f32::MAX`) or underflow-to-`+Inf` (e.g.
  /// subnormal magnitudes).
  ///
  /// # Errors
  /// [`Error::EmbeddingDimMismatch`] if `s.len() != `[`EMBEDDING_DIM`];
  /// [`Error::NonFiniteEmbedding`] on any non-finite component;
  /// [`Error::EmbeddingZero`] if the input has zero magnitude.
  pub fn from_slice_normalizing(s: &[f32]) -> Result<Self> {
    if s.len() != EMBEDDING_DIM {
      return Err(Error::EmbeddingDimMismatch {
        expected: EMBEDDING_DIM,
        got: s.len(),
      });
    }
    for (i, &v) in s.iter().enumerate() {
      if !v.is_finite() {
        return Err(Error::NonFiniteEmbedding { component_index: i });
      }
    }
    // f64 accumulation: for any finite f32 (|x| ≤ ~3.4e38), x² ≤ ~1.16e77 and
    // 384 terms sum to at most ~4.5e79, well inside f64's ~1.8e308 range.
    let norm_sq_f64: f64 = s.iter().map(|&x| (x as f64) * (x as f64)).sum();
    if norm_sq_f64 == 0.0 {
      return Err(Error::EmbeddingZero);
    }
    let inv_norm_f64 = 1.0_f64 / norm_sq_f64.sqrt();
    // Multiply per-component in f64 then cast: casting inv_norm to f32 first
    // would overflow to +Inf for subnormal-magnitude inputs.
    let mut inner = [0.0f32; EMBEDDING_DIM];
    for (out, &v) in inner.iter_mut().zip(s.iter()) {
      *out = ((v as f64) * inv_norm_f64) as f32;
    }
    Ok(Self { inner })
  }

  /// Module-internal constructor for producers whose upstream guard already
  /// validated unit-norm within `NORM_BUDGET`. Bypasses re-normalization.
  #[allow(dead_code)]
  pub(crate) fn from_array_trusted_unit_norm(arr: [f32; EMBEDDING_DIM]) -> Self {
    debug_assert!({
      let n: f32 = arr.iter().map(|x| x * x).sum();
      (n - 1.0).abs() <= NORM_BUDGET
    });
    Self { inner: arr }
  }

  /// Inner product. For two unit vectors this equals [`Self::cosine`] to fp32
  /// ULP.
  pub fn dot(&self, other: &Embedding) -> f32 {
    self
      .inner
      .iter()
      .zip(other.inner.iter())
      .map(|(a, b)| a * b)
      .sum()
  }

  /// Cosine similarity. For unit vectors, equivalent to [`Self::dot`].
  pub fn cosine(&self, other: &Embedding) -> f32 {
    self.dot(other)
  }

  /// Approximate equality — max-abs metric. `true` iff
  /// `(self − other).max_abs() ≤ tol` (inclusive, so `is_close(self, 0.0)` is
  /// always true).
  pub fn is_close(&self, other: &Embedding, tol: f32) -> bool {
    self
      .inner
      .iter()
      .zip(other.inner.iter())
      .map(|(a, b)| (a - b).abs())
      .fold(0.0f32, f32::max)
      <= tol
  }

  /// Approximate equality — semantic (cosine) metric. `true` iff
  /// `1 − cosine(other) ≤ tol`, computed as `0.5·‖a − b‖² ≤ tol` to avoid
  /// catastrophic cancellation near identity (valid because both operands are
  /// unit-norm to fp32 ULP).
  pub fn is_close_cosine(&self, other: &Embedding, tol: f32) -> bool {
    let sq: f32 = self
      .inner
      .iter()
      .zip(other.inner.iter())
      .map(|(a, b)| {
        let d = a - b;
        d * d
      })
      .sum();
    (sq * 0.5) <= tol
  }
}

impl AsRef<[f32]> for Embedding {
  #[inline]
  fn as_ref(&self) -> &[f32] {
    self.as_slice()
  }
}

impl Deref for Embedding {
  type Target = [f32];

  #[inline]
  fn deref(&self) -> &[f32] {
    &self.inner
  }
}

impl fmt::Debug for Embedding {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "Embedding {{ dim: {}, head: [{:.4}, {:.4}, {:.4}, ..] }}",
      self.dim(),
      self.inner[0],
      self.inner[1],
      self.inner[2],
    )
  }
}

/// Scans a raw model-output projection — the copied CoreML tensor, before it is
/// normalized into an [`Embedding`] — for the first non-finite (NaN/±∞)
/// component, classifying it as MODEL corruption ([`Error::NonFiniteOutput`]).
/// This is the counterpart to the caller-data corruption
/// ([`Error::NonFiniteEmbedding`]) that [`Embedding::from_slice_normalizing`]
/// raises for a caller's own slice: the embedder runs this on the model output
/// *before* normalizing, so a NaN the runtime produced is reported as
/// model-output corruption rather than mislabeled as caller-supplied embedding
/// data (the workspace convention — it mirrors the CLAP tower's identically
/// shaped `check_finite_output`). Extracted so the classification is
/// hermetically testable without a loaded model.
///
/// # Errors
/// [`Error::NonFiniteOutput`] carrying the flat index of the first non-finite
/// component.
pub(crate) fn check_finite_output(values: &[f32]) -> Result<()> {
  if let Some(index) = values.iter().position(|v| !v.is_finite()) {
    return Err(Error::NonFiniteOutput { index });
  }
  Ok(())
}

#[cfg(test)]
mod tests;
