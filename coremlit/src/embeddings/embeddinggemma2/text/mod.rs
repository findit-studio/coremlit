//! The text part: the seven functions of the text package and their set, the
//! inputs they are fed, and the artifacts and instances that run them.

mod embedder;
pub(crate) mod layout;

pub use embedder::{Framing, TextArtifacts, TextEmbedder};

/// One function of `EmbeddingGemma2Text.mlpackage`.
///
/// Each `embed_S` embeds ONE sequence of at most `S` ids, padded on the right.
/// `pack_256` embeds up to eight sequences whose ids total at most 256 rows, in
/// one call. All seven share one set of weights.
///
/// [`Self::as_str`] is the function's name as Core ML loads it, and the
/// `Display`, the serde spelling and the document spelling are that same word.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, derive_more::Display)]
#[display("{}", self.as_str())]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum TextFunction {
  /// `embed_32`: one sequence of at most 32 ids.
  #[cfg_attr(feature = "serde", serde(rename = "embed_32"))]
  Embed32,
  /// `embed_48`: one sequence of at most 48 ids.
  #[cfg_attr(feature = "serde", serde(rename = "embed_48"))]
  Embed48,
  /// `embed_64`: one sequence of at most 64 ids.
  #[cfg_attr(feature = "serde", serde(rename = "embed_64"))]
  Embed64,
  /// `embed_128`: one sequence of at most 128 ids. The package's default
  /// function.
  #[cfg_attr(feature = "serde", serde(rename = "embed_128"))]
  Embed128,
  /// `embed_256`: one sequence of at most 256 ids.
  #[cfg_attr(feature = "serde", serde(rename = "embed_256"))]
  Embed256,
  /// `embed_512`: one sequence of at most 512 ids, the model's ceiling.
  #[cfg_attr(feature = "serde", serde(rename = "embed_512"))]
  Embed512,
  /// `pack_256`: up to eight sequences of at most 256 ids in all, each in its
  /// own attention block.
  #[cfg_attr(feature = "serde", serde(rename = "pack_256"))]
  Pack256,
}

impl TextFunction {
  /// The seven functions, in the order the package declares them.
  pub const ALL: [Self; 7] = [
    Self::Embed32,
    Self::Embed48,
    Self::Embed64,
    Self::Embed128,
    Self::Embed256,
    Self::Embed512,
    Self::Pack256,
  ];

  /// The function's name in the package: `"embed_32"` … `"embed_512"`,
  /// `"pack_256"`.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Embed32 => "embed_32",
      Self::Embed48 => "embed_48",
      Self::Embed64 => "embed_64",
      Self::Embed128 => "embed_128",
      Self::Embed256 => "embed_256",
      Self::Embed512 => "embed_512",
      Self::Pack256 => "pack_256",
    }
  }

  /// The rows of the function's `inputs_embeds`: `S` for `embed_S`, and 256
  /// for `pack_256`.
  #[inline]
  pub const fn tokens(&self) -> usize {
    match self {
      Self::Embed32 => 32,
      Self::Embed48 => 48,
      Self::Embed64 => 64,
      Self::Embed128 => 128,
      Self::Embed256 => 256,
      Self::Embed512 => 512,
      Self::Pack256 => 256,
    }
  }

  /// This function's bit in a [`TextFunctions`] set.
  #[inline]
  const fn bit(self) -> u8 {
    1 << self as u8
  }
}

/// A set of [`TextFunction`]s: the functions a text instance warms at
/// `prewarm`.
///
/// Its `Display`, `Debug` and serde forms list the members in package order,
/// `[embed_32,pack_256]`; a document that names one function twice is refused
/// by name rather than read as the set it would collapse to.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextFunctions(u8);

impl TextFunctions {
  /// The empty set.
  #[inline]
  pub const fn empty() -> Self {
    Self(0)
  }

  /// All seven functions.
  #[inline]
  pub const fn all() -> Self {
    Self(0b0111_1111)
  }

  /// This set with `function` added.
  #[must_use]
  #[inline]
  pub const fn with(self, function: TextFunction) -> Self {
    Self(self.0 | function.bit())
  }

  /// Whether `function` is in the set.
  #[inline]
  pub const fn contains(&self, function: TextFunction) -> bool {
    self.0 & function.bit() != 0
  }

  /// Whether the set is empty.
  #[inline]
  pub const fn is_empty(&self) -> bool {
    self.0 == 0
  }

  /// How many functions the set holds.
  #[inline]
  pub const fn len(&self) -> usize {
    self.0.count_ones() as usize
  }

  /// The members, in package order.
  pub fn iter(&self) -> impl Iterator<Item = TextFunction> + '_ {
    TextFunction::ALL
      .into_iter()
      .filter(move |function| self.contains(*function))
  }
}

impl FromIterator<TextFunction> for TextFunctions {
  fn from_iter<I: IntoIterator<Item = TextFunction>>(iter: I) -> Self {
    iter.into_iter().fold(Self::empty(), Self::with)
  }
}

impl core::fmt::Debug for TextFunctions {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_set().entries(self.iter()).finish()
  }
}

/// `[` the members' names in package order, joined by `,`, `]`; the empty set
/// prints `[]`.
impl core::fmt::Display for TextFunctions {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str("[")?;
    for (index, function) in self.iter().enumerate() {
      if index > 0 {
        f.write_str(",")?;
      }
      f.write_str(function.as_str())?;
    }
    f.write_str("]")
  }
}

#[cfg(feature = "serde")]
impl serde::Serialize for TextFunctions {
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;

    let mut seq = serializer.serialize_seq(Some(self.len()))?;
    for function in self.iter() {
      seq.serialize_element(&function)?;
    }
    seq.end()
  }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for TextFunctions {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    struct Members;

    impl<'de> serde::de::Visitor<'de> for Members {
      type Value = TextFunctions;

      fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("a list of distinct text function names")
      }

      fn visit_seq<A: serde::de::SeqAccess<'de>>(
        self,
        mut seq: A,
      ) -> Result<Self::Value, A::Error> {
        let mut set = TextFunctions::empty();
        while let Some(function) = seq.next_element::<TextFunction>()? {
          if set.contains(function) {
            return Err(serde::de::Error::custom(format_args!(
              "the text function `{function}` is listed twice"
            )));
          }
          set = set.with(function);
        }
        Ok(set)
      }
    }

    deserializer.deserialize_seq(Members)
  }
}

#[cfg(test)]
mod tests;
