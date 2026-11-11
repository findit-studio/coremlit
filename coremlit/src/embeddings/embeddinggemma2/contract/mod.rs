//! A compiled contract for a JSON document, and the walk that checks a
//! document against it while it parses.
//!
//! The walk reads every value as the type its field expects and builds no
//! `serde_json::Value`, so a document's shape costs nothing to hold, whatever
//! it is. What it allocates is bounded by the document's own text and by the
//! contract: a string or a key that carries an escape is unescaped once into
//! serde_json's scratch, an object's entries are tracked in a list as long as
//! the contract's, and a difference quotes at most 64 bytes of what the
//! document holds. The first difference in the contract's order is named by
//! its dotted path.

use core::fmt::{self, Write as _};

use serde::de::{self, DeserializeSeed, Deserializer, IgnoredAny, MapAccess, SeqAccess, Visitor};

use super::error::{Excerpt, excerpt};

/// What one field of a contract must hold.
#[derive(Debug)]
pub(crate) enum Expect {
  /// A non-negative integer.
  Uint(u64),
  /// A number, compared exactly as the f64 it parses to.
  Float(f64),
  /// A boolean.
  Bool(bool),
  /// `null`.
  Null,
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
  /// An object with at least these keys; any other key's value is read past
  /// and held nowhere.
  Open(&'static [(&'static str, &'static str, Expect)]),
}

/// The first difference under a field: its dotted path, the contract's value
/// and the document's, at most 64 bytes of it.
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
pub(crate) struct Render<'c>(pub(crate) &'c Expect);

impl fmt::Display for Render<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self.0 {
      Expect::Uint(n) => write!(f, "{n}"),
      Expect::Float(x) => write!(f, "{}", JsonFloat(*x)),
      Expect::Bool(b) => write!(f, "{b}"),
      Expect::Null => f.write_str("null"),
      Expect::Str(s) => write!(f, "{}", JsonStr(s)),
      Expect::Uints(ns) => {
        f.write_char('[')?;
        for (i, n) in ns.iter().enumerate() {
          if i > 0 {
            f.write_char(',')?;
          }
          write!(f, "{n}")?;
        }
        f.write_char(']')
      }
      Expect::Strs(ss) => {
        f.write_char('[')?;
        for (i, s) in ss.iter().enumerate() {
          if i > 0 {
            f.write_char(',')?;
          }
          write!(f, "{}", JsonStr(s))?;
        }
        f.write_char(']')
      }
      Expect::Elements(items) => {
        f.write_char('[')?;
        for (i, (_, item)) in items.iter().enumerate() {
          if i > 0 {
            f.write_char(',')?;
          }
          write!(f, "{}", Render(item))?;
        }
        f.write_char(']')
      }
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

/// Checks one value of a document against `expect`, the field at `path`: the
/// first difference under it, or `None`.
#[derive(Clone, Copy)]
pub(crate) struct Walk<'c> {
  pub(crate) expect: &'c Expect,
  pub(crate) path: &'static str,
}

impl Walk<'_> {
  fn differs(self, actual: String) -> Option<Difference> {
    Some(Difference {
      path: self.path,
      expected: Render(self.expect).to_string(),
      actual,
    })
  }

  fn leaf(self, holds: bool, actual: impl fmt::Display) -> Option<Difference> {
    if holds {
      None
    } else {
      self.differs(excerpt(actual))
    }
  }
}

impl<'de> DeserializeSeed<'de> for Walk<'_> {
  type Value = Option<Difference>;

  fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
    deserializer.deserialize_any(self)
  }
}

impl<'de> Visitor<'de> for Walk<'_> {
  type Value = Option<Difference>;

  fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("a JSON value")
  }

  fn visit_bool<E: de::Error>(self, v: bool) -> Result<Self::Value, E> {
    Ok(self.leaf(matches!(self.expect, Expect::Bool(b) if *b == v), v))
  }

  fn visit_i64<E: de::Error>(self, v: i64) -> Result<Self::Value, E> {
    let holds = match *self.expect {
      Expect::Uint(n) => u64::try_from(v) == Ok(n),
      Expect::Float(x) => v as f64 == x,
      _ => false,
    };
    Ok(self.leaf(holds, v))
  }

  fn visit_u64<E: de::Error>(self, v: u64) -> Result<Self::Value, E> {
    let holds = match *self.expect {
      Expect::Uint(n) => v == n,
      Expect::Float(x) => v as f64 == x,
      _ => false,
    };
    Ok(self.leaf(holds, v))
  }

  fn visit_f64<E: de::Error>(self, v: f64) -> Result<Self::Value, E> {
    let holds = matches!(*self.expect, Expect::Float(x) if x == v);
    Ok(self.leaf(holds, JsonFloat(v)))
  }

  fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
    let holds = matches!(*self.expect, Expect::Str(s) if s == v);
    Ok(self.leaf(holds, JsonStr(v)))
  }

  fn visit_unit<E: de::Error>(self) -> Result<Self::Value, E> {
    Ok(self.leaf(matches!(self.expect, Expect::Null), "null"))
  }

  fn visit_none<E: de::Error>(self) -> Result<Self::Value, E> {
    self.visit_unit()
  }

  fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Self::Value, A::Error> {
    match *self.expect {
      Expect::Uints(_) | Expect::Strs(_) => {
        let mut shown = Excerpt::new();
        let _ = shown.write_char('[');
        let (mut index, mut holds) = (0usize, true);
        while let Some(matched) = seq.next_element_seed(Item {
          expect: self.expect,
          index,
          shown: &mut shown,
        })? {
          holds &= matched;
          index += 1;
        }
        let _ = shown.write_char(']');
        holds &= match *self.expect {
          Expect::Uints(ns) => index == ns.len(),
          Expect::Strs(ss) => index == ss.len(),
          _ => false,
        };
        Ok(if holds {
          None
        } else {
          self.differs(shown.finish())
        })
      }
      Expect::Elements(items) => {
        let (mut first, mut count) = (None, 0usize);
        loop {
          let verdict = match items.get(count) {
            Some((path, expect)) => match seq.next_element_seed(Walk { expect, path })? {
              Some(verdict) => verdict,
              None => break,
            },
            None => match seq.next_element::<IgnoredAny>()? {
              Some(_) => None,
              None => break,
            },
          };
          if first.is_none() {
            first = verdict;
          }
          count += 1;
        }
        Ok(first.or_else(|| {
          (count != items.len()).then(|| Difference {
            path: self.path,
            expected: Render(self.expect).to_string(),
            actual: format!("an array of {count} elements"),
          })
        }))
      }
      _ => {
        let mut shown = Excerpt::new();
        sketch_seq(&mut seq, &mut shown)?;
        Ok(self.differs(shown.finish()))
      }
    }
  }

  fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Self::Value, A::Error> {
    let (entries, closed) = match *self.expect {
      Expect::Object(entries) => (entries, true),
      Expect::Open(entries) => (entries, false),
      _ => {
        let mut shown = Excerpt::new();
        sketch_map(&mut map, &mut shown)?;
        return Ok(self.differs(shown.finish()));
      }
    };
    // One slot per entry of the contract: the contract's size, not the
    // document's. A key given twice is judged by its last value.
    let mut found: Vec<Option<Option<Difference>>> = Vec::with_capacity(entries.len());
    found.resize_with(entries.len(), || None);
    let mut extra = None;
    while let Some(key) = map.next_key_seed(Key(entries))? {
      match key {
        Ok(index) => {
          let (_, path, expect) = &entries[index];
          found[index] = Some(map.next_value_seed(Walk { expect, path })?);
        }
        Err(name) => {
          if closed && extra.is_none() {
            extra = Some(name);
          }
          map.next_value::<IgnoredAny>()?;
        }
      }
    }
    for ((_, path, expect), seen) in entries.iter().zip(found) {
      match seen {
        None => {
          return Ok(Some(Difference {
            path,
            expected: Render(expect).to_string(),
            actual: "absent".to_owned(),
          }));
        }
        Some(Some(difference)) => return Ok(Some(difference)),
        Some(None) => {}
      }
    }
    Ok(extra.map(|name| Difference {
      path: self.path,
      expected: format!("exactly the keys [{}]", Keys(entries)),
      actual: format!("also the key {name}"),
    }))
  }
}

/// An object's key: its entry's index in the contract, or, for a key the
/// contract does not know, its quoted excerpt.
struct Key<'c>(&'c [(&'static str, &'static str, Expect)]);

impl<'de> DeserializeSeed<'de> for Key<'_> {
  type Value = Result<usize, String>;

  fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<Self::Value, D::Error> {
    deserializer.deserialize_str(self)
  }
}

impl Visitor<'_> for Key<'_> {
  type Value = Result<usize, String>;

  fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("an object's key")
  }

  fn visit_str<E: de::Error>(self, v: &str) -> Result<Self::Value, E> {
    Ok(
      self
        .0
        .iter()
        .position(|(key, _, _)| *key == v)
        .ok_or_else(|| excerpt(format_args!("{v:?}"))),
    )
  }
}

/// One element of an array of scalars: compared with the contract's element
/// at `index` and shown in the array's excerpt.
struct Item<'c, 's> {
  expect: &'c Expect,
  index: usize,
  shown: &'s mut Excerpt,
}

impl Item<'_, '_> {
  fn show(&mut self, value: impl fmt::Display) {
    if self.index > 0 {
      let _ = self.shown.write_char(',');
    }
    let _ = write!(self.shown, "{value}");
  }
}

impl<'de> DeserializeSeed<'de> for Item<'_, '_> {
  type Value = bool;

  fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<bool, D::Error> {
    deserializer.deserialize_any(self)
  }
}

impl<'de> Visitor<'de> for Item<'_, '_> {
  type Value = bool;

  fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("a JSON value")
  }

  fn visit_bool<E: de::Error>(mut self, v: bool) -> Result<bool, E> {
    self.show(v);
    Ok(false)
  }

  fn visit_i64<E: de::Error>(mut self, v: i64) -> Result<bool, E> {
    self.show(v);
    Ok(
      matches!(*self.expect, Expect::Uints(ns) if u64::try_from(v).is_ok_and(|v| ns.get(self.index) == Some(&v))),
    )
  }

  fn visit_u64<E: de::Error>(mut self, v: u64) -> Result<bool, E> {
    self.show(v);
    Ok(matches!(*self.expect, Expect::Uints(ns) if ns.get(self.index) == Some(&v)))
  }

  fn visit_f64<E: de::Error>(mut self, v: f64) -> Result<bool, E> {
    self.show(JsonFloat(v));
    Ok(false)
  }

  fn visit_str<E: de::Error>(mut self, v: &str) -> Result<bool, E> {
    self.show(JsonStr(v));
    Ok(matches!(*self.expect, Expect::Strs(ss) if ss.get(self.index).is_some_and(|s| *s == v)))
  }

  fn visit_unit<E: de::Error>(mut self) -> Result<bool, E> {
    self.show("null");
    Ok(false)
  }

  fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<bool, A::Error> {
    if self.index > 0 {
      let _ = self.shown.write_char(',');
    }
    sketch_seq(&mut seq, self.shown)?;
    Ok(false)
  }

  fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<bool, A::Error> {
    if self.index > 0 {
      let _ = self.shown.write_char(',');
    }
    sketch_map(&mut map, self.shown)?;
    Ok(false)
  }
}

/// Writes any value's JSON into an excerpt, a separator first when `lead`
/// holds one; once the excerpt is full, the value is read on and dropped.
struct Sketch<'s> {
  shown: &'s mut Excerpt,
  lead: Option<char>,
}

impl Sketch<'_> {
  fn show(self, value: impl fmt::Display) {
    if let Some(lead) = self.lead {
      let _ = self.shown.write_char(lead);
    }
    let _ = write!(self.shown, "{value}");
  }
}

impl<'de> DeserializeSeed<'de> for Sketch<'_> {
  type Value = ();

  fn deserialize<D: Deserializer<'de>>(self, deserializer: D) -> Result<(), D::Error> {
    deserializer.deserialize_any(self)
  }
}

impl<'de> Visitor<'de> for Sketch<'_> {
  type Value = ();

  fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    f.write_str("a JSON value")
  }

  fn visit_bool<E: de::Error>(self, v: bool) -> Result<(), E> {
    self.show(v);
    Ok(())
  }

  fn visit_i64<E: de::Error>(self, v: i64) -> Result<(), E> {
    self.show(v);
    Ok(())
  }

  fn visit_u64<E: de::Error>(self, v: u64) -> Result<(), E> {
    self.show(v);
    Ok(())
  }

  fn visit_f64<E: de::Error>(self, v: f64) -> Result<(), E> {
    self.show(JsonFloat(v));
    Ok(())
  }

  fn visit_str<E: de::Error>(self, v: &str) -> Result<(), E> {
    self.show(JsonStr(v));
    Ok(())
  }

  fn visit_unit<E: de::Error>(self) -> Result<(), E> {
    self.show("null");
    Ok(())
  }

  fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<(), A::Error> {
    if let Some(lead) = self.lead {
      let _ = self.shown.write_char(lead);
    }
    sketch_seq(&mut seq, self.shown)
  }

  fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<(), A::Error> {
    if let Some(lead) = self.lead {
      let _ = self.shown.write_char(lead);
    }
    sketch_map(&mut map, self.shown)
  }
}

/// An array's JSON into `shown`, its elements read through.
fn sketch_seq<'de, A: SeqAccess<'de>>(seq: &mut A, shown: &mut Excerpt) -> Result<(), A::Error> {
  let _ = shown.write_char('[');
  let mut lead = None;
  while let Some(()) = seq.next_element_seed(Sketch {
    shown: &mut *shown,
    lead,
  })? {
    lead = Some(',');
  }
  let _ = shown.write_char(']');
  Ok(())
}

/// An object's JSON into `shown`, its entries read through.
fn sketch_map<'de, A: MapAccess<'de>>(map: &mut A, shown: &mut Excerpt) -> Result<(), A::Error> {
  let _ = shown.write_char('{');
  let mut lead = None;
  while let Some(()) = map.next_key_seed(Sketch {
    shown: &mut *shown,
    lead,
  })? {
    let _ = shown.write_char(':');
    map.next_value_seed(Sketch {
      shown: &mut *shown,
      lead: None,
    })?;
    lead = Some(',');
  }
  let _ = shown.write_char('}');
  Ok(())
}
