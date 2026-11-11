//! A compiled contract for a JSON document, and its comparison with the
//! document's typed parse.
//!
//! The document is parsed by serde's own derived readers into typed structs
//! ([`typed_object!`]): an object the kit owns refuses a key it does not
//! declare, and every number is read by serde's own impl, so a build that
//! turns on serde_json's `arbitrary_precision` reads the document the same. A
//! key that is missing reads as absent rather than failing the parse, so the
//! comparison can name it. [`compare`] then walks the contract in its order and
//! names the first field, by its dotted path, that is absent or holds another
//! value.
//!
//! The other keys of an [`Expect::Open`] object are skipped by serde, not
//! validated.

use core::fmt::{self, Write as _};

use serde::{Deserialize, Deserializer};

use super::error::excerpt;

/// What one field of a contract must hold.
#[derive(Debug)]
pub(crate) enum Expect {
  /// A non-negative integer.
  Uint(u64),
  /// A number, compared exactly as the f64 it parses to.
  Float(f64),
  /// A boolean.
  Bool(bool),
  /// A string, byte for byte.
  Str(&'static str),
  /// An array of non-negative integers.
  Uints(&'static [u64]),
  /// An array of strings.
  Strs(&'static [&'static str]),
  /// An array of exactly these values: `(dotted path, value)`.
  Elements(&'static [(&'static str, Expect)]),
  /// An object with exactly these keys: `(key, dotted path, value)`.
  Object(&'static [(&'static str, &'static str, Expect)]),
  /// An object with at least these keys; its other keys are read past.
  Open(&'static [(&'static str, &'static str, Expect)]),
}

/// The first difference under a field: its dotted path, the contract's value
/// and the document's, at most 64 bytes of each.
#[derive(Debug)]
pub(crate) struct Difference {
  pub(crate) path: &'static str,
  pub(crate) expected: String,
  pub(crate) actual: String,
}

impl fmt::Display for Difference {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    write!(
      f,
      "at `{}`: expected {}, got {}",
      self.path, self.expected, self.actual
    )
  }
}

/// What the typed parse holds at one field.
#[derive(Clone, Copy)]
pub(crate) enum Actual<'a> {
  /// The key is missing.
  Absent,
  Bool(bool),
  Uint(u64),
  Float(f64),
  Str(&'a str),
  Uints(&'a [u64]),
  Strs(&'a [String]),
  Elements(&'a dyn Elements),
  Object(&'a dyn Fields),
}

impl<'a> Actual<'a> {
  /// A field serde read as `Some` where its key was present.
  pub(crate) fn of<T: AsActual>(field: &'a Option<T>) -> Self {
    field.as_ref().map_or(Self::Absent, AsActual::as_actual)
  }
}

/// A typed value as the comparison sees it.
pub(crate) trait AsActual {
  fn as_actual(&self) -> Actual<'_>;
}

impl AsActual for bool {
  fn as_actual(&self) -> Actual<'_> {
    Actual::Bool(*self)
  }
}

impl AsActual for u64 {
  fn as_actual(&self) -> Actual<'_> {
    Actual::Uint(*self)
  }
}

impl AsActual for f64 {
  fn as_actual(&self) -> Actual<'_> {
    Actual::Float(*self)
  }
}

impl AsActual for String {
  fn as_actual(&self) -> Actual<'_> {
    Actual::Str(self)
  }
}

impl AsActual for Vec<u64> {
  fn as_actual(&self) -> Actual<'_> {
    Actual::Uints(self)
  }
}

impl AsActual for Vec<String> {
  fn as_actual(&self) -> Actual<'_> {
    Actual::Strs(self)
  }
}

/// A typed object: each declared field by its key.
pub(crate) trait Fields {
  /// The keys the type declares.
  fn keys(&self) -> &'static [&'static str];
  /// The field at `key`; [`Actual::Absent`] where the key was missing or is
  /// not declared.
  fn field(&self, key: &str) -> Actual<'_>;
}

/// A typed array of objects.
pub(crate) trait Elements {
  fn count(&self) -> usize;
  fn element(&self, index: usize) -> Actual<'_>;
}

/// Reads a field that is present as its own type. With `#[serde(default)]`,
/// a missing key is `None`, and a `null` is refused like any value of another
/// type: absent and null stay apart.
pub(crate) fn some<'de, D: Deserializer<'de>, T: Deserialize<'de>>(
  deserializer: D,
) -> Result<Option<T>, D::Error> {
  T::deserialize(deserializer).map(Some)
}

/// Declares a closed typed object of a contract: a derived struct that
/// refuses any key it does not declare, each field `Some` where its key is
/// present, with its [`Fields`] and [`AsActual`].
macro_rules! typed_object {
  (
    $(#[$meta:meta])*
    struct $name:ident { $( $key:literal => $field:ident: $ty:ty ),+ $(,)? }
  ) => {
    #[derive(serde::Deserialize)]
    #[serde(deny_unknown_fields)]
    $(#[$meta])*
    struct $name {
      $(
        #[serde(
          rename = $key,
          default,
          deserialize_with = "crate::embeddings::embeddinggemma2::contract::some"
        )]
        $field: Option<$ty>,
      )+
    }

    $crate::embeddings::embeddinggemma2::contract::object_fields!($name { $( $key => $field ),+ });
  };
}

/// [`typed_object!`] for an open object: the keys it does not declare are
/// skipped by serde, not validated.
macro_rules! open_object {
  (
    $(#[$meta:meta])*
    struct $name:ident { $( $key:literal => $field:ident: $ty:ty ),+ $(,)? }
  ) => {
    #[derive(serde::Deserialize)]
    $(#[$meta])*
    struct $name {
      $(
        #[serde(
          rename = $key,
          default,
          deserialize_with = "crate::embeddings::embeddinggemma2::contract::some"
        )]
        $field: Option<$ty>,
      )+
    }

    $crate::embeddings::embeddinggemma2::contract::object_fields!($name { $( $key => $field ),+ });
  };
}

/// A typed object's [`Fields`] and [`AsActual`].
macro_rules! object_fields {
  ($name:ident { $( $key:literal => $field:ident ),+ }) => {
    impl $crate::embeddings::embeddinggemma2::contract::Fields for $name {
      fn keys(&self) -> &'static [&'static str] {
        &[$($key),+]
      }

      fn field(&self, key: &str) -> $crate::embeddings::embeddinggemma2::contract::Actual<'_> {
        match key {
          $( $key => $crate::embeddings::embeddinggemma2::contract::Actual::of(&self.$field), )+
          _ => $crate::embeddings::embeddinggemma2::contract::Actual::Absent,
        }
      }
    }

    impl $crate::embeddings::embeddinggemma2::contract::AsActual for $name {
      fn as_actual(&self) -> $crate::embeddings::embeddinggemma2::contract::Actual<'_> {
        $crate::embeddings::embeddinggemma2::contract::Actual::Object(self)
      }
    }
  };
}

pub(crate) use {object_fields, open_object, typed_object};

/// The first difference between the contract's `expect` and the parse's
/// `actual`, the field at `path`, in the contract's order: a field that is
/// absent, a value that differs, or a key a closed object holds and the
/// contract does not.
pub(crate) fn compare(
  expect: &Expect,
  path: &'static str,
  actual: Actual<'_>,
) -> Option<Difference> {
  let differs = |actual: String| {
    Some(Difference {
      path,
      expected: excerpt(Render(expect)),
      actual,
    })
  };
  match (expect, actual) {
    (_, Actual::Absent) => differs("absent".to_owned()),
    (Expect::Object(entries) | Expect::Open(entries), Actual::Object(object)) => entries
      .iter()
      .find_map(|(key, path, expect)| compare(expect, path, object.field(key)))
      .or_else(|| {
        if !matches!(expect, Expect::Object(_)) {
          return None;
        }
        let extra = object.keys().iter().find(|key| {
          !matches!(object.field(key), Actual::Absent)
            && !entries.iter().any(|(known, _, _)| known == *key)
        })?;
        differs(format!("also the key {}", JsonStr(extra)))
      }),
    (Expect::Elements(items), Actual::Elements(elements)) => {
      if elements.count() != items.len() {
        return differs(excerpt(Shown(actual)));
      }
      items
        .iter()
        .enumerate()
        .find_map(|(index, (path, expect))| compare(expect, path, elements.element(index)))
    }
    (Expect::Uint(n), Actual::Uint(v)) if *n == v => None,
    (Expect::Float(x), Actual::Float(v)) if *x == v => None,
    (Expect::Bool(b), Actual::Bool(v)) if *b == v => None,
    (Expect::Str(s), Actual::Str(v)) if *s == v => None,
    (Expect::Uints(ns), Actual::Uints(vs)) if *ns == vs => None,
    (Expect::Strs(ss), Actual::Strs(vs))
      if ss.len() == vs.len() && ss.iter().zip(vs).all(|(s, v)| s == v) =>
    {
      None
    }
    (_, actual) => differs(excerpt(Shown(actual))),
  }
}

/// One step of the path at which a parse refused: an object's key or an
/// array's index.
pub(crate) enum Step<'a> {
  Key(&'a str),
  Index(usize),
}

/// The deepest field of `contract` that holds the place at `steps` from the
/// top level: its dotted path and what the contract holds there. A step the
/// contract does not hold ends the descent, so a key the contract does not
/// know names the object that carries it.
pub(crate) fn field_at<'s>(
  contract: &'static Expect,
  steps: impl IntoIterator<Item = Step<'s>>,
) -> (&'static str, &'static Expect) {
  let mut here: (&'static str, &'static Expect) = ("", contract);
  for step in steps {
    let next = match (here.1, step) {
      (Expect::Object(entries) | Expect::Open(entries), Step::Key(key)) => entries
        .iter()
        .find(|(known, _, _)| *known == key)
        .map(|(_, path, expect)| (*path, expect)),
      (Expect::Elements(items), Step::Index(index)) => {
        items.get(index).map(|(path, expect)| (*path, expect))
      }
      _ => None,
    };
    match next {
      Some(next) => here = next,
      None => break,
    }
  }
  here
}

/// What the contract holds at a field, for a refusal the parse raised there:
/// a value as JSON, an object as the keys it carries.
pub(crate) struct Wanted<'c>(pub(crate) &'c Expect);

impl fmt::Display for Wanted<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self.0 {
      Expect::Object(entries) => write!(f, "exactly the keys [{}]", Keys(entries)),
      Expect::Open(entries) => write!(f, "at least the keys [{}]", Keys(entries)),
      expect => write!(f, "{}", Render(expect)),
    }
  }
}

/// A string as a JSON literal.
struct JsonStr<'a>(&'a str);

impl fmt::Display for JsonStr<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_char('"')?;
    for c in self.0.chars() {
      match c {
        '"' => f.write_str("\\\"")?,
        '\\' => f.write_str("\\\\")?,
        '\n' => f.write_str("\\n")?,
        '\r' => f.write_str("\\r")?,
        '\t' => f.write_str("\\t")?,
        c if u32::from(c) < 0x20 => write!(f, "\\u{:04x}", u32::from(c))?,
        c => f.write_char(c)?,
      }
    }
    f.write_char('"')
  }
}

/// A float as serde_json writes it.
struct JsonFloat(f64);

impl fmt::Display for JsonFloat {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match serde_json::Number::from_f64(self.0) {
      Some(number) => write!(f, "{number}"),
      None => f.write_str("null"),
    }
  }
}

/// The contract's value at a field, as JSON.
struct Render<'c>(&'c Expect);

impl fmt::Display for Render<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self.0 {
      Expect::Uint(n) => write!(f, "{n}"),
      Expect::Float(x) => write!(f, "{}", JsonFloat(*x)),
      Expect::Bool(b) => write!(f, "{b}"),
      Expect::Str(s) => write!(f, "{}", JsonStr(s)),
      Expect::Uints(ns) => list(f, ns.iter(), |f, n| write!(f, "{n}")),
      Expect::Strs(ss) => list(f, ss.iter(), |f, s| write!(f, "{}", JsonStr(s))),
      Expect::Elements(items) => list(f, items.iter(), |f, (_, item)| {
        write!(f, "{}", Render(item))
      }),
      Expect::Object(entries) | Expect::Open(entries) => {
        f.write_char('{')?;
        for (i, (key, _, value)) in entries.iter().enumerate() {
          if i > 0 {
            f.write_char(',')?;
          }
          write!(f, "{}:{}", JsonStr(key), Render(value))?;
        }
        f.write_char('}')
      }
    }
  }
}

/// The parse's value at a field, as JSON; an object or an array of objects
/// by what it is.
struct Shown<'a>(Actual<'a>);

impl fmt::Display for Shown<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self.0 {
      Actual::Absent => f.write_str("absent"),
      Actual::Bool(b) => write!(f, "{b}"),
      Actual::Uint(n) => write!(f, "{n}"),
      Actual::Float(x) => write!(f, "{}", JsonFloat(x)),
      Actual::Str(s) => write!(f, "{}", JsonStr(s)),
      Actual::Uints(ns) => list(f, ns.iter(), |f, n| write!(f, "{n}")),
      Actual::Strs(ss) => list(f, ss.iter(), |f, s| write!(f, "{}", JsonStr(s))),
      Actual::Elements(elements) => write!(f, "an array of {} objects", elements.count()),
      Actual::Object(_) => f.write_str("an object"),
    }
  }
}

/// `[a,b,…]`, each item written by `each`.
fn list<T>(
  f: &mut fmt::Formatter<'_>,
  items: impl Iterator<Item = T>,
  mut each: impl FnMut(&mut fmt::Formatter<'_>, T) -> fmt::Result,
) -> fmt::Result {
  f.write_char('[')?;
  for (i, item) in items.enumerate() {
    if i > 0 {
      f.write_char(',')?;
    }
    each(f, item)?;
  }
  f.write_char(']')
}

/// An object's keys as the contract lists them: `"a", "b"`.
struct Keys<'c>(&'c [(&'static str, &'static str, Expect)]);

impl fmt::Display for Keys<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    for (i, (key, _, _)) in self.0.iter().enumerate() {
      if i > 0 {
        f.write_str(", ")?;
      }
      write!(f, "{key:?}")?;
    }
    Ok(())
  }
}
