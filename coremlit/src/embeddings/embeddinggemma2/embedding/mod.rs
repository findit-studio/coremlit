//! The unit-norm vector every modality embeds to, and its Matryoshka widths.

/// The model's embedding width.
pub const EMBEDDING_DIM: usize = 768;

/// A width an [`Embedding`] may have: one of the model's Matryoshka widths.
///
/// `Width<N>` is [`Matryoshka`] for exactly `N` ∈ {768, 512, 256, 128}, so a
/// vector of any other width cannot be built: `Embedding<300>` names a type no
/// constructor produces, and `truncate::<300>()` does not compile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Width<const N: usize>(());

/// The Matryoshka widths: implemented for [`Width<768>`], [`Width<512>`],
/// [`Width<256>`] and [`Width<128>`], and sealed, so no other width exists.
pub trait Matryoshka: sealed::Sealed {}

mod sealed {
  pub trait Sealed {}
}

macro_rules! matryoshka_widths {
  ($($n:literal),+) => {$(
    impl sealed::Sealed for Width<$n> {}
    impl Matryoshka for Width<$n> {}
  )+};
}

matryoshka_widths!(768, 512, 256, 128);

/// A unit-norm (L2) embedding of width `N`, in f32: the model's 768-d vector,
/// or a Matryoshka truncation of it.
///
/// Every modality embeds into this one space: text, audio windows and images
/// alike come out of the text functions, so their vectors compare directly.
/// Queries and documents must share a width, and the width is the type's, so
/// a 256-wide query against a 768-wide column does not compile.
///
/// The vector the model emits is checked finite and re-normalized in f64. No
/// `PartialEq` (it carries a model's floats) and no serde, as the crate's other
/// embedding types.
#[derive(Clone)]
pub struct Embedding<const N: usize = EMBEDDING_DIM> {
  inner: [f32; N],
}

impl<const N: usize> Embedding<N>
where
  Width<N>: Matryoshka,
{
  /// The vector's width, `N`.
  #[inline]
  pub const fn dim(&self) -> usize {
    N
  }

  /// The components.
  #[inline]
  pub const fn as_slice(&self) -> &[f32] {
    &self.inner
  }

  /// The components, copied into a new `Vec`.
  #[inline]
  pub fn to_vec(&self) -> Vec<f32> {
    self.inner.to_vec()
  }

  /// The dot product with `other`, accumulated in f64.
  pub fn dot(&self, other: &Self) -> f32 {
    self
      .inner
      .iter()
      .zip(&other.inner)
      .map(|(a, b)| f64::from(*a) * f64::from(*b))
      .sum::<f64>() as f32
  }

  /// The cosine similarity with `other`. Both vectors are unit-norm, so it is
  /// their dot product.
  #[inline]
  pub fn cosine(&self, other: &Self) -> f32 {
    self.dot(other)
  }

  /// `wide` scaled to unit length, narrowed to f32; `None` when its length is
  /// zero or not finite.
  fn from_wide(wide: &[f64; N]) -> Option<Self> {
    let norm = wide.iter().map(|v| v * v).sum::<f64>().sqrt();
    if !(norm.is_finite() && norm > 0.0) {
      return None;
    }
    Some(Self {
      inner: core::array::from_fn(|i| (wide[i] / norm) as f32),
    })
  }
}

impl Embedding<EMBEDDING_DIM> {
  /// The first `M` components, re-normalized in f64: the model's Matryoshka
  /// truncation. `M` must be a Matryoshka width.
  ///
  /// ```
  /// # use coremlit::embeddings::embeddinggemma2::Embedding;
  /// # fn widths(v: &Embedding) {
  /// let short: Embedding<256> = v.truncate();
  /// assert_eq!(short.dim(), 256);
  /// # }
  /// ```
  ///
  /// The prefix of a unit vector of the model's is never all zero; if it
  /// were, it would have no direction, and it is returned as zeros.
  pub fn truncate<const M: usize>(&self) -> Embedding<M>
  where
    Width<M>: Matryoshka,
  {
    let wide: [f64; M] = core::array::from_fn(|i| f64::from(self.inner[i]));
    Embedding::from_wide(&wide).unwrap_or(Embedding { inner: [0.0; M] })
  }
}

impl<const N: usize> AsRef<[f32]> for Embedding<N>
where
  Width<N>: Matryoshka,
{
  #[inline]
  fn as_ref(&self) -> &[f32] {
    self.as_slice()
  }
}

impl<const N: usize> core::fmt::Debug for Embedding<N>
where
  Width<N>: Matryoshka,
{
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(
      f,
      "Embedding {{ dim: {N}, head: [{:.4}, {:.4}, {:.4}, ..] }}",
      self.inner[0], self.inner[1], self.inner[2],
    )
  }
}

/// So a text's windows fold through `windit::aggregate`, as granite's do.
impl<const N: usize> windit::windowed::Vector for Embedding<N>
where
  Width<N>: Matryoshka,
{
  type Scalar = f32;

  fn as_slice(&self) -> &[f32] {
    &self.inner
  }

  fn from_unnormalized(v: &[f64]) -> Result<Self, windit::WinditError> {
    let wide: &[f64; N] = v.try_into().map_err(|_| windit::WinditError::DimMismatch {
      got: v.len(),
      expected: N,
    })?;
    Self::from_wide(wide).ok_or(windit::WinditError::NonFinite)
  }
}

#[cfg(test)]
mod tests;
