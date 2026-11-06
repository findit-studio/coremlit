//! The vision part: the budgets of the vision package, their set, the resize
//! road, the artifacts an image is prepared with, and the instances that
//! encode it.

mod encoder;
mod prepare;

pub use encoder::ImageEncoder;
pub use prepare::{PreparedImage, VisionArtifacts};

/// A vision token budget: one function of `EmbeddingGemma2Vision.mlpackage`.
///
/// `vision_T` takes `9 × T` patches of 16 × 16 pixels and pools them, three by
/// three, into `T` image tokens.
///
/// Two spellings, each for its own reader:
/// - [`Self::as_str`] is the function's name as Core ML loads it,
///   `"vision_140"`;
/// - the `Display`, the serde form and the document form are the token count,
///   `140`, which is how a derivation term spells the budget (`budget=140`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[non_exhaustive]
pub enum VisionBudget {
  /// 70 tokens from 630 patches (`vision_70`), the package's default function.
  Tokens70,
  /// 140 tokens from 1,260 patches (`vision_140`): this kit's default, and the
  /// original model's per-frame budget for video.
  Tokens140,
  /// 280 tokens from 2,520 patches (`vision_280`): the original model's image
  /// default.
  Tokens280,
}

impl VisionBudget {
  /// The three budgets, smallest first.
  pub const ALL: [Self; 3] = [Self::Tokens70, Self::Tokens140, Self::Tokens280];

  /// The function's name in the package: `"vision_70"`, `"vision_140"`,
  /// `"vision_280"`.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Tokens70 => "vision_70",
      Self::Tokens140 => "vision_140",
      Self::Tokens280 => "vision_280",
    }
  }

  /// The image tokens the function emits: 70, 140 or 280.
  #[inline]
  pub const fn tokens(&self) -> usize {
    match self {
      Self::Tokens70 => 70,
      Self::Tokens140 => 140,
      Self::Tokens280 => 280,
    }
  }

  /// The patches the function takes: `9 × tokens`.
  #[inline]
  pub const fn patches(&self) -> usize {
    9 * self.tokens()
  }

  /// The budget of `tokens` image tokens, or `None` for a count the package
  /// does not ship.
  #[inline]
  pub const fn from_tokens(tokens: usize) -> Option<Self> {
    match tokens {
      70 => Some(Self::Tokens70),
      140 => Some(Self::Tokens140),
      280 => Some(Self::Tokens280),
      _ => None,
    }
  }

  /// This budget's bit in a [`VisionBudgets`] set.
  #[inline]
  const fn bit(self) -> u8 {
    1 << self as u8
  }
}

/// The token count: `70`, `140` or `280`.
impl core::fmt::Display for VisionBudget {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    write!(f, "{}", self.tokens())
  }
}

#[cfg(feature = "serde")]
impl serde::Serialize for VisionBudget {
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    serializer.serialize_u16(self.tokens() as u16)
  }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for VisionBudget {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    let tokens = <u64 as serde::Deserialize>::deserialize(deserializer)?;
    usize::try_from(tokens)
      .ok()
      .and_then(Self::from_tokens)
      .ok_or_else(|| {
        serde::de::Error::custom(format_args!(
          "unknown vision budget {tokens}: expected one of 70, 140, 280"
        ))
      })
  }
}

/// A set of [`VisionBudget`]s: the `vision_T` functions an image instance
/// loads.
///
/// Its `Display`, `Debug` and serde forms list the token counts smallest
/// first, `[70,140]`; a document that names one budget twice is refused by name.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct VisionBudgets(u8);

impl VisionBudgets {
  /// The empty set.
  #[inline]
  pub const fn empty() -> Self {
    Self(0)
  }

  /// The set holding `budget` alone.
  #[inline]
  pub const fn only(budget: VisionBudget) -> Self {
    Self(budget.bit())
  }

  /// All three budgets.
  #[inline]
  pub const fn all() -> Self {
    Self(0b111)
  }

  /// This set with `budget` added.
  #[must_use]
  #[inline]
  pub const fn with(self, budget: VisionBudget) -> Self {
    Self(self.0 | budget.bit())
  }

  /// Whether `budget` is in the set.
  #[inline]
  pub const fn contains(&self, budget: VisionBudget) -> bool {
    self.0 & budget.bit() != 0
  }

  /// Whether the set is empty.
  #[inline]
  pub const fn is_empty(&self) -> bool {
    self.0 == 0
  }

  /// How many budgets the set holds.
  #[inline]
  pub const fn len(&self) -> usize {
    self.0.count_ones() as usize
  }

  /// The members, smallest first.
  pub fn iter(&self) -> impl Iterator<Item = VisionBudget> + '_ {
    VisionBudget::ALL
      .into_iter()
      .filter(move |budget| self.contains(*budget))
  }
}

impl FromIterator<VisionBudget> for VisionBudgets {
  fn from_iter<I: IntoIterator<Item = VisionBudget>>(iter: I) -> Self {
    iter.into_iter().fold(Self::empty(), Self::with)
  }
}

impl core::fmt::Debug for VisionBudgets {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.debug_set().entries(self.iter()).finish()
  }
}

/// `[` the members' token counts, smallest first, joined by `,`, `]`; the
/// empty set prints `[]`.
impl core::fmt::Display for VisionBudgets {
  fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
    f.write_str("[")?;
    for (index, budget) in self.iter().enumerate() {
      if index > 0 {
        f.write_str(",")?;
      }
      write!(f, "{budget}")?;
    }
    f.write_str("]")
  }
}

#[cfg(feature = "serde")]
impl serde::Serialize for VisionBudgets {
  fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
    use serde::ser::SerializeSeq;

    let mut seq = serializer.serialize_seq(Some(self.len()))?;
    for budget in self.iter() {
      seq.serialize_element(&budget)?;
    }
    seq.end()
  }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for VisionBudgets {
  fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
    struct Members;

    impl<'de> serde::de::Visitor<'de> for Members {
      type Value = VisionBudgets;

      fn expecting(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str("a list of distinct vision budgets")
      }

      fn visit_seq<A: serde::de::SeqAccess<'de>>(
        self,
        mut seq: A,
      ) -> Result<Self::Value, A::Error> {
        let mut set = VisionBudgets::empty();
        while let Some(budget) = seq.next_element::<VisionBudget>()? {
          if set.contains(budget) {
            return Err(serde::de::Error::custom(format_args!(
              "the vision budget {budget} is listed twice"
            )));
          }
          set = set.with(budget);
        }
        Ok(set)
      }
    }

    deserializer.deserialize_seq(Members)
  }
}

/// How an image is resized to its budget's geometry before patching.
///
/// The one road today is Hugging Face's. A later road (Apple's scaler) joins
/// as a variant of its own, never as the default, and is spelled in the
/// derivation terms like this one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, derive_more::Display)]
#[display("{}", self.as_str())]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
#[non_exhaustive]
pub enum ResizeRoad {
  /// `Gemma4ImageProcessor`'s resize: antialiased bicubic on 8-bit RGB (PIL's
  /// `BICUBIC`, Keys `a = −0.5`), through pixon's `CatmullRom` resampler. The
  /// default.
  HfBicubic,
}

impl ResizeRoad {
  /// Stable snake_case name of the road: `"hf_bicubic"`.
  #[inline]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::HfBicubic => "hf_bicubic",
    }
  }
}

#[cfg(test)]
mod tests;
