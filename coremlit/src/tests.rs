//! The in-lib half of the model-gate report (#61).
//!
//! Most of this crate's model gates are NOT integration tests: they are
//! `#[ignore]`d unit tests inside the pipeline modules — more of them than
//! every `tests/` binary holds put together — and CI reaches them through
//! three `model-tests` shards: whisper's two `@all` groups, which build and run
//! the lib target alongside every integration one, and the granite and speaker
//! shards' `@lib` ones. The `align` gates reach no shard at all, because
//! alignkit has no MODELS_LOCK table; ci.yml's matrix carries the per-kit
//! ledger, counts included. They are skipped by the same
//! silence, so they get the same report; see
//! `coremlit/tests/support/model_gate_report.rs` for the mechanism.
//!
//! That module lives under `tests/` because every other caller is a test
//! binary, and this one `#[path]`-hops to it rather than keeping a second copy
//! — the `tests/support/coremlit_dir.rs` convention, one level further. The hop
//! is `#[cfg(test)]`, so a published crate never compiles it.

use std::path::PathBuf;

#[path = "../tests/support/model_gate_report.rs"]
mod model_gate_report;

// The same `#[path]` hop, for the anchor every in-lib `models_dir()` and the
// crate-wide variant guard resolve against. FOUND by searching upward for the
// `[workspace]` manifest, never counted in `../` hops — see its module doc.
// The re-export is `pub(crate)` because the `#[ignore]`d gates that need it are
// scattered across `src/**/tests.rs`, which reach it as `crate::tests::…`.
#[path = "../tests/support/workspace_root.rs"]
#[allow(dead_code)]
mod workspace_root;
pub(crate) use workspace_root::models_root;

/// `<workspace>/Models/<sub>` unless `var` overrides it — the fallback every
/// in-lib `models_dir()` in this crate resolves. `sub` is empty for whisper,
/// whose gates read the `Models/` root itself.
fn root(var: &str, sub: &str) -> PathBuf {
  std::env::var_os(var).map_or_else(
    || {
      let models = models_root();
      if sub.is_empty() {
        models
      } else {
        models.join(sub)
      }
    },
    PathBuf::from,
  )
}

/// Reports how many of the library's own unit tests are `#[ignore]`d model
/// gates that did not run, and whether the models roots they read are on disk.
///
/// The roots are named per feature because the gates are: a bare
/// `default = []` build compiles no pipeline, so it has no model gates and
/// claims no roots. `cfg!` rather than `#[cfg]` on the elements, so the vector
/// is used mutably on every combination — including that empty one.
///
/// Today's in-lib gates span exactly these four kits (whisper, align, speaker,
/// granite); the `clap`/`siglip`/`ced`/`vad` gates are all in `tests/`, where
/// their own `common/mod.rs` names their root. A new in-lib gate under one of
/// those would still be COUNTED — the count is libtest's, not this list's — it
/// would simply have no root named beside it until a line is added here.
#[test]
fn model_gate_report() {
  let mut roots: Vec<(&str, PathBuf)> = Vec::new();
  if cfg!(feature = "whisper") {
    roots.push(("WHISPERKIT_TEST_MODELS", root("WHISPERKIT_TEST_MODELS", "")));
  }
  if cfg!(feature = "align") {
    roots.push((
      "ALIGNKIT_TEST_MODELS",
      root("ALIGNKIT_TEST_MODELS", "alignkit"),
    ));
  }
  if cfg!(feature = "speaker") {
    roots.push((
      "SPEAKERKIT_TEST_MODELS",
      root("SPEAKERKIT_TEST_MODELS", "speakerkit"),
    ));
    roots.push((
      "ARGMAX_TEST_MODELS",
      root("ARGMAX_TEST_MODELS", "argmax-speakerkit"),
    ));
  }
  if cfg!(feature = "granite") {
    roots.push((
      "EMBEDKIT_TEST_MODELS",
      root("EMBEDKIT_TEST_MODELS", "embedkit-granite"),
    ));
  }
  model_gate_report::report(&roots);
}

/// A counting global allocator, and the probe the allocation-ORDERING
/// falsifiers read it through.
///
/// `#[cfg(test)]` by construction — this whole module is — so only the lib test
/// binary carries it. Every other target (each `tests/` binary, each bench,
/// every doctest, and the published library) links this crate without
/// `cfg(test)` and keeps the platform allocator untouched.
///
/// # Why an allocator and not a resident-set reading
///
/// The property under test is "this path did not ASK for the buffer", which is
/// what `GlobalAlloc` sees. `ru_maxrss` is a process-wide high-water mark that
/// never falls, so under libtest's parallel threads it reports whatever the
/// rest of the suite peaked at; and a `calloc`'d gigabyte that is never read
/// commits no resident pages at all, so RSS cannot even see the allocation this
/// measures.
///
/// # Thread-local, deliberately
///
/// libtest runs each `#[test]` on its own thread, so per-thread counters
/// measure the test that installed the probe and nothing the rest of the suite
/// is doing beside it. They are `Cell<usize>` with const initialisation and no
/// destructor, so reading them from inside `GlobalAlloc` allocates nothing and
/// cannot re-enter the allocator.
///
/// Memory freed on a thread that did not allocate it drives `LIVE` toward zero
/// (it saturates there), which can only UNDER-report a peak — never invent one.
pub(crate) mod alloc_probe {
  use std::{
    alloc::{GlobalAlloc, Layout, System},
    cell::Cell,
  };

  thread_local! {
    /// Bytes this thread has asked for and not yet released.
    static LIVE: Cell<usize> = const { Cell::new(0) };
    /// High-water mark of `LIVE` since the last [`measure`] began.
    static PEAK: Cell<usize> = const { Cell::new(0) };
    /// Bytes this thread has asked for in total, releases included.
    static TOTAL: Cell<usize> = const { Cell::new(0) };
  }

  #[inline]
  fn record_alloc(bytes: usize) {
    let _ = TOTAL.try_with(|t| t.set(t.get().saturating_add(bytes)));
    let _ = LIVE.try_with(|l| {
      let now = l.get().saturating_add(bytes);
      l.set(now);
      let _ = PEAK.try_with(|p| {
        if now > p.get() {
          p.set(now);
        }
      });
    });
  }

  #[inline]
  fn record_free(bytes: usize) {
    let _ = LIVE.try_with(|l| l.set(l.get().saturating_sub(bytes)));
  }

  /// Forwards every request to [`System`] unchanged and counts the sizes.
  struct Counting;

  // SAFETY: every method forwards to `System` — the same platform allocator the
  // test binary would otherwise install — with the caller's pointer and layout
  // passed through unchanged, so the pointers returned, their validity, and the
  // alloc/dealloc pairing are exactly `System`'s and satisfy `GlobalAlloc`'s
  // contract. The bookkeeping added around each call touches only const-init,
  // `Drop`-free thread-local `Cell<usize>`s, which cannot allocate and so cannot
  // re-enter this allocator.
  unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
      // SAFETY: `layout` is the caller's own, forwarded unchanged.
      let ptr = unsafe { System.alloc(layout) };
      if !ptr.is_null() {
        record_alloc(layout.size());
      }
      ptr
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
      // SAFETY: `layout` is the caller's own, forwarded unchanged.
      let ptr = unsafe { System.alloc_zeroed(layout) };
      if !ptr.is_null() {
        record_alloc(layout.size());
      }
      ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
      record_free(layout.size());
      // SAFETY: `ptr` and `layout` are the caller's own, forwarded unchanged —
      // this allocator returned `ptr` from `System` for that very layout.
      unsafe { System.dealloc(ptr, layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
      // SAFETY: `ptr`, `layout` and `new_size` are the caller's own, forwarded
      // unchanged — this allocator returned `ptr` from `System` for `layout`.
      let new_ptr = unsafe { System.realloc(ptr, layout, new_size) };
      if !new_ptr.is_null() {
        record_free(layout.size());
        record_alloc(new_size);
      }
      new_ptr
    }
  }

  #[global_allocator]
  static ALLOCATOR: Counting = Counting;

  /// What a [`measure`]d closure asked the allocator for, in bytes.
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  pub(crate) struct Allocated {
    /// High-water mark of the bytes held at once.
    pub(crate) peak: usize,
    /// Bytes requested in total, whether or not they were released again.
    pub(crate) total: usize,
  }

  /// Runs `f` and reports the allocation it performed ON THIS THREAD.
  ///
  /// Anything `f` returns is still live when the measurement is taken, so a
  /// buffer it hands back counts in both figures.
  pub(crate) fn measure<T>(f: impl FnOnce() -> T) -> (T, Allocated) {
    let live = LIVE.with(Cell::get);
    let total = TOTAL.with(Cell::get);
    PEAK.with(|p| p.set(live));
    let out = f();
    (
      out,
      Allocated {
        peak: PEAK.with(Cell::get).saturating_sub(live),
        total: TOTAL.with(Cell::get).saturating_sub(total),
      },
    )
  }

  #[test]
  fn the_probe_sees_a_buffer_it_is_pointed_at() {
    // A self-check, so a probe that silently measured nothing could not make an
    // ordering falsifier pass by reporting zero for every path.
    let (v, a) = measure(|| vec![0u8; 4_000_000]);
    assert_eq!(v.len(), 4_000_000);
    assert!(
      a.total >= 4_000_000 && a.peak >= 4_000_000,
      "the probe must see a 4 MB allocation it wraps: {a:?}"
    );
    // ...and a released buffer still counts in `total` while leaving `peak`
    // where it was, which is what separates "asked for it" from "held it".
    let (_, released) = measure(|| drop(vec![0u8; 8_000_000]));
    assert!(released.total >= 8_000_000, "{released:?}");
  }
}

// ══════════════════════════════════════════════════════════════════════
// The crate-wide enum-variant SHAPE guard
// ══════════════════════════════════════════════════════════════════════

/// The house rule this module enforces over every `.rs` file in the workspace
/// (`rust-type-conventions/references/enums-errors.md`): **variants are UNIT
/// or NEWTYPE only, never struct-shaped**. When several fields belong to one
/// variant they are extracted into a named payload struct that the variant
/// wraps.
///
/// It asserts the SHAPE, not the absence of a brace. A brace-counting check
/// would pass `Variant(A, B)`, which breaks the rule's *purpose* — the reason
/// a variant carries at most one unnamed field is that there must be exactly
/// one nameable payload type to hand back — while reading as proof that it
/// could not. Both illegal shapes are rejected here.
///
/// This runs over source text rather than over types because the property is
/// about how the code is WRITTEN, and because the point of a guard is to hold
/// for code that does not exist yet.
mod variant_shape {
  use std::{
    fmt,
    path::{Path, PathBuf},
  };

  /// How one variant carries its payload.
  #[derive(Debug, Clone, Copy, PartialEq, Eq)]
  pub enum Shape {
    /// `Variant`, or `Variant = <discriminant>`. Allowed.
    Unit,
    /// `Variant(T)` — exactly one unnamed field. Allowed.
    Newtype,
    /// `Variant(..)` with a field count that is not 1. Forbidden: a 2-tuple
    /// has no single nameable payload type, and `Variant()` is a unit variant
    /// written the long way.
    Tuple(usize),
    /// `Variant { .. }` — the struct shape the rule forbids outright.
    Struct,
  }

  impl Shape {
    /// Whether the house rule permits this shape.
    pub const fn is_legal(self) -> bool {
      matches!(self, Self::Unit | Self::Newtype)
    }
  }

  impl fmt::Display for Shape {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
      match self {
        Self::Unit => f.write_str("unit"),
        Self::Newtype => f.write_str("newtype"),
        Self::Tuple(n) => write!(f, "{n}-tuple"),
        Self::Struct => f.write_str("struct-shaped"),
      }
    }
  }

  /// One declared variant: where it is, what encloses it, and its shape.
  #[derive(Debug, Clone, PartialEq, Eq)]
  pub struct Variant {
    pub line: usize,
    pub enum_name: String,
    pub name: String,
    pub shape: Shape,
  }

  const fn is_ident_start(b: u8) -> bool {
    b.is_ascii_alphabetic() || b == b'_' || b >= 0x80
  }

  const fn is_ident_continue(b: u8) -> bool {
    is_ident_start(b) || b.is_ascii_digit()
  }

  /// Replaces every comment body and every string/char literal body with
  /// spaces, keeping byte length and newlines so offsets and line numbers
  /// stay exact. Lifetimes (`'a`, `'static`) are NOT literals and are left
  /// alone — mistaking one for an unterminated char literal would blank the
  /// rest of the file and make the whole guard fail open.
  pub fn blank_noise(src: &str) -> String {
    let b = src.as_bytes();
    let mut out: Vec<u8> = Vec::with_capacity(b.len());
    let mut i = 0;
    // Blanks bytes `i..end`, keeping newlines.
    macro_rules! blank_to {
      ($end:expr) => {{
        let end = $end;
        while i < end {
          out.push(if b[i] == b'\n' { b'\n' } else { b' ' });
          i += 1;
        }
      }};
    }
    while i < b.len() {
      // Identifiers first, so `raw_value` is not read as a raw string and
      // `b` / `c` / `r` prefixes are attached to the literal that follows.
      if is_ident_start(b[i]) {
        let start = i;
        while i < b.len() && is_ident_continue(b[i]) {
          i += 1;
        }
        let word = &src[start..i];
        let prefixed_raw =
          matches!(word, "r" | "br" | "cr") && i < b.len() && (b[i] == b'"' || b[i] == b'#');
        let prefixed_str = matches!(word, "b" | "c") && i < b.len() && b[i] == b'"';
        out.extend_from_slice(word.as_bytes());
        if prefixed_raw {
          let hashes = {
            let h = i;
            while i < b.len() && b[i] == b'#' {
              i += 1;
            }
            i - h
          };
          if i < b.len() && b[i] == b'"' {
            out.resize(out.len() + hashes + 1, b' ');
            i += 1;
            let close = format!("\"{}", "#".repeat(hashes));
            let end = src[i..].find(&close).map_or(b.len(), |p| i + p);
            blank_to!(end);
            blank_to!((end + close.len()).min(b.len()));
          } else {
            out.resize(out.len() + hashes, b'#');
          }
          continue;
        }
        if !prefixed_str {
          continue;
        }
        // fall through into the plain-string arm below
      }
      // Line comment.
      if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'/' {
        let end = src[i..].find('\n').map_or(b.len(), |p| i + p);
        blank_to!(end);
        continue;
      }
      // Block comment, nested.
      if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
        let mut depth = 0usize;
        while i < b.len() {
          if b[i] == b'/' && i + 1 < b.len() && b[i + 1] == b'*' {
            depth += 1;
            blank_to!(i + 2);
            continue;
          }
          if b[i] == b'*' && i + 1 < b.len() && b[i + 1] == b'/' {
            depth -= 1;
            blank_to!(i + 2);
            if depth == 0 {
              break;
            }
            continue;
          }
          blank_to!(i + 1);
        }
        continue;
      }
      // Plain string literal (also the tail of a `b"`/`c"` prefix).
      if b[i] == b'"' {
        out.push(b' ');
        i += 1;
        while i < b.len() {
          if b[i] == b'\\' && i + 1 < b.len() {
            blank_to!(i + 2);
            continue;
          }
          if b[i] == b'"' {
            out.push(b' ');
            i += 1;
            break;
          }
          blank_to!(i + 1);
        }
        continue;
      }
      // Char literal vs lifetime.
      if b[i] == b'\'' {
        let rest = &src[i + 1..];
        let mut chars = rest.chars();
        let literal_end = match chars.next() {
          Some('\\') => rest.find('\'').map(|p| i + 1 + p + 1),
          Some(c) => {
            let w = c.len_utf8();
            (rest.as_bytes().get(w) == Some(&b'\'')).then_some(i + 1 + w + 1)
          }
          None => None,
        };
        match literal_end {
          Some(end) => blank_to!(end),
          None => {
            out.push(b'\'');
            i += 1;
          }
        }
        continue;
      }
      out.push(b[i]);
      i += 1;
    }
    String::from_utf8(out).expect("blanking only ever emits ASCII or whole original bytes")
  }

  /// Skips a balanced `open`/`close` run starting at `bytes[from] == open`,
  /// returning the index just past the matching close.
  fn skip_balanced(bytes: &[u8], from: usize, open: u8, close: u8) -> usize {
    let mut depth = 0usize;
    let mut i = from;
    while i < bytes.len() {
      if bytes[i] == open {
        depth += 1;
      } else if bytes[i] == close {
        depth -= 1;
        if depth == 0 {
          return i + 1;
        }
      }
      i += 1;
    }
    bytes.len()
  }

  /// Counts a tuple variant's fields: commas at paren depth 1 with no open
  /// angle bracket, so `Variant(BTreeMap<String, f64>)` is ONE field, not two.
  /// `->` is consumed whole so a function-pointer field does not unbalance
  /// the angle count.
  fn tuple_arity(body: &str) -> usize {
    let b = body.as_bytes();
    let (mut paren, mut angle, mut square, mut brace) = (0usize, 0usize, 0usize, 0usize);
    let mut fields = 0usize;
    let mut saw_content = false;
    let mut i = 0;
    while i < b.len() {
      match b[i] {
        b'-' if i + 1 < b.len() && b[i + 1] == b'>' => {
          i += 2;
          continue;
        }
        b'(' => paren += 1,
        b')' => paren -= 1,
        b'[' => square += 1,
        b']' => square = square.saturating_sub(1),
        b'{' => brace += 1,
        b'}' => brace = brace.saturating_sub(1),
        b'<' => angle += 1,
        b'>' => angle = angle.saturating_sub(1),
        b',' if paren == 1 && angle == 0 && square == 0 && brace == 0 => {
          if saw_content {
            fields += 1;
          }
          saw_content = false;
          i += 1;
          continue;
        }
        c if !c.is_ascii_whitespace() && paren >= 1 => saw_content = true,
        _ => {}
      }
      i += 1;
    }
    fields + usize::from(saw_content)
  }

  /// Every variant declared in one file's (already blanked) source.
  pub fn variants_in(src: &str) -> Vec<Variant> {
    let blanked = blank_noise(src);
    let b = blanked.as_bytes();
    let mut out = Vec::new();
    let mut i = 0;
    while i < b.len() {
      // `enum` as a whole token.
      if !(blanked[i..].starts_with("enum")
        && (i == 0 || !is_ident_continue(b[i - 1]))
        && b.get(i + 4).is_none_or(|&c| !is_ident_continue(c)))
      {
        i += 1;
        continue;
      }
      let mut j = i + 4;
      while j < b.len() && b[j].is_ascii_whitespace() {
        j += 1;
      }
      let name_start = j;
      while j < b.len() && is_ident_continue(b[j]) {
        j += 1;
      }
      let enum_name = blanked[name_start..j].to_string();
      let Some(open) = blanked[j..].find('{').map(|p| j + p) else {
        i += 4;
        continue;
      };
      let close = skip_balanced(b, open, b'{', b'}');
      let body_start = open + 1;
      let body_end = close.saturating_sub(1);
      // Split the body into entries at top-level commas.
      let body = &blanked[body_start..body_end];
      let bb = body.as_bytes();
      let (mut paren, mut brace, mut square) = (0usize, 0usize, 0usize);
      let mut start = 0usize;
      let mut entries: Vec<(usize, &str)> = Vec::new();
      for (k, &c) in bb.iter().enumerate() {
        match c {
          b'(' => paren += 1,
          b')' => paren = paren.saturating_sub(1),
          b'{' => brace += 1,
          b'}' => brace = brace.saturating_sub(1),
          b'[' => square += 1,
          b']' => square = square.saturating_sub(1),
          b',' if paren == 0 && brace == 0 && square == 0 => {
            entries.push((body_start + start, &body[start..k]));
            start = k + 1;
          }
          _ => {}
        }
      }
      entries.push((body_start + start, &body[start..]));
      for (offset, entry) in entries {
        // Strip leading attributes, then read the variant name.
        let eb = entry.as_bytes();
        let mut k = 0usize;
        loop {
          while k < eb.len() && eb[k].is_ascii_whitespace() {
            k += 1;
          }
          if k < eb.len() && eb[k] == b'#' {
            let br = entry[k..].find('[').map(|p| k + p);
            match br {
              Some(br) => k = skip_balanced(eb, br, b'[', b']'),
              None => break,
            }
          } else {
            break;
          }
        }
        if k >= eb.len() || !is_ident_start(eb[k]) {
          continue;
        }
        let vs = k;
        while k < eb.len() && is_ident_continue(eb[k]) {
          k += 1;
        }
        let name = entry[vs..k].to_string();
        while k < eb.len() && eb[k].is_ascii_whitespace() {
          k += 1;
        }
        let shape = match eb.get(k) {
          Some(b'{') => Shape::Struct,
          Some(b'(') => match tuple_arity(&entry[k..]) {
            1 => Shape::Newtype,
            n => Shape::Tuple(n),
          },
          _ => Shape::Unit,
        };
        let line = blanked[..offset + vs].matches('\n').count() + 1;
        out.push(Variant {
          line,
          enum_name: enum_name.clone(),
          name,
          shape,
        });
      }
      i = close;
    }
    out
  }

  /// The workspace root — the tree the rule holds over.
  ///
  /// FOUND, not counted: `super::workspace_root` walks up to the manifest that
  /// declares `[workspace]`, so no package can move out from under this.
  pub fn source_root() -> PathBuf {
    super::workspace_root::workspace_root()
  }

  /// Every `.rs` file under `dir`, skipping `target/`, `.git/` and `Models/`:
  /// build output, history, and the gitignored artifact-download tree. None of
  /// the three is this repository's source, and a fetched `Models/` can hold
  /// tens of thousands of files some vendor happened to publish.
  pub fn rust_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
      let entries = std::fs::read_dir(&d).unwrap_or_else(|e| panic!("read {}: {e}", d.display()));
      for entry in entries {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
          if path
            .file_name()
            .is_some_and(|n| n == "target" || n == ".git" || n == "Models")
          {
            continue;
          }
          stack.push(path);
        } else if path.extension().is_some_and(|e| e == "rs") {
          out.push(path);
        }
      }
    }
    out.sort();
    out
  }
}

/// **The sweep's finish line.** No enum variant anywhere in the workspace is
/// struct-shaped, and none carries more than one unnamed field — the house
/// rule, enforced for code that has not been written yet rather than
/// re-asserted per module.
///
/// Replaces the per-module grandfather census that lived in
/// `audio/speaker/error/tests.rs`: that one read a single file and looked for
/// a brace, so it could neither see another module nor a `Variant(A, B)`.
#[test]
fn no_enum_in_the_workspace_has_a_struct_shaped_or_multi_field_variant() {
  use variant_shape::{rust_files, source_root, variants_in};

  let root = source_root();
  assert!(
    root.is_dir(),
    "the workspace root must be reachable at {}",
    root.display()
  );
  let files = rust_files(&root);
  let mut variants = 0usize;
  let mut enums = std::collections::BTreeSet::new();
  let mut illegal: Vec<String> = Vec::new();
  for file in &files {
    let src = std::fs::read_to_string(file).expect("read source");
    for v in variants_in(&src) {
      variants += 1;
      enums.insert((file.clone(), v.enum_name.clone()));
      if !v.shape.is_legal() {
        illegal.push(format!(
          "{}:{}  {}::{} is {}",
          file.strip_prefix(&root).unwrap_or(file).display(),
          v.line,
          v.enum_name,
          v.name,
          v.shape
        ));
      }
    }
  }
  // A scan that reached nothing would pass vacuously, which is the failure
  // mode of every checker that reconciles only against itself. Floors well
  // under the real tree, so they catch a broken walk, not ordinary growth.
  assert!(files.len() >= 200, "only {} .rs files found", files.len());
  assert!(enums.len() >= 60, "only {} enums found", enums.len());
  assert!(variants >= 250, "only {variants} variants found");
  assert!(
    illegal.is_empty(),
    "variants must be UNIT or a NEWTYPE of exactly one payload \
     (rust-type-conventions: never struct-shaped, never a 2-tuple):\n{}",
    illegal.join("\n")
  );
}

/// The guard's own detector, run against known-bad and known-tricky input.
/// A checker that has only ever been seen to pass is not a checker: these are
/// the shapes it must FLAG, and the ones it must not.
#[test]
fn the_shape_scanner_flags_exactly_the_illegal_shapes() {
  use variant_shape::{Shape, variants_in};

  let shapes = |src: &str| -> Vec<(String, Shape)> {
    variants_in(src)
      .into_iter()
      .map(|v| (v.name, v.shape))
      .collect()
  };

  // ── the two illegal shapes ──
  assert_eq!(
    shapes("enum E { A { x: u32 } }"),
    vec![("A".to_string(), Shape::Struct)]
  );
  assert_eq!(
    shapes("enum E { A {} }"),
    vec![("A".to_string(), Shape::Struct)]
  );
  assert_eq!(
    shapes("enum E { A(u32, u32) }"),
    vec![("A".to_string(), Shape::Tuple(2))]
  );
  assert_eq!(
    shapes("enum E { A() }"),
    vec![("A".to_string(), Shape::Tuple(0))]
  );
  assert_eq!(
    shapes("enum E { A(u8, u8, u8) }"),
    vec![("A".to_string(), Shape::Tuple(3))]
  );

  // ── the legal ones, including every comma that must NOT be counted ──
  assert_eq!(
    shapes(
      "enum E { A, B(u32), C = 3, D(BTreeMap<String, f64>), F(Result<A, B>), \
       G(fn(u8, u8) -> u8), H(&'static str), I([u8; 4]), J(Vec<Vec<u8>>), K(u32,) }"
    ),
    vec![
      ("A".to_string(), Shape::Unit),
      ("B".to_string(), Shape::Newtype),
      ("C".to_string(), Shape::Unit),
      ("D".to_string(), Shape::Newtype),
      ("F".to_string(), Shape::Newtype),
      ("G".to_string(), Shape::Newtype),
      ("H".to_string(), Shape::Newtype),
      ("I".to_string(), Shape::Newtype),
      ("J".to_string(), Shape::Newtype),
      ("K".to_string(), Shape::Newtype),
    ]
  );

  // ── attributes, including a format string holding braces and a comma ──
  assert_eq!(
    shapes("enum E { #[error(\"a { } b, c\")] A(u32), #[cfg(feature = \"x\")] B }"),
    vec![
      ("A".to_string(), Shape::Newtype),
      ("B".to_string(), Shape::Unit)
    ]
  );

  // ── text that only LOOKS like a declaration is not one ──
  for hidden in [
    "let s = \"enum E { A { x: u32 } }\";",
    "// enum E { A { x: u32 } }",
    "/// enum E { A { x: u32 } }",
    "/* enum E { A { x: u32 } } */",
    "/* /* enum E { A { x: u32 } } */ */",
    "let s = r#\"enum E { A { x: u32 } }\"#;",
    "let s = r\"enum E { A { x: u32 } }\";",
  ] {
    assert!(
      variants_in(hidden).is_empty(),
      "must not read a declaration out of {hidden}"
    );
  }

  // A lifetime is not an unterminated char literal: mistaking one would blank
  // the rest of the file and every later violation with it.
  assert_eq!(
    shapes("enum A<'a> { X(&'a str) } enum B { Y { z: u8 } }"),
    vec![
      ("X".to_string(), Shape::Newtype),
      ("Y".to_string(), Shape::Struct)
    ]
  );
  // ...and a real char literal still is one.
  assert!(variants_in("let c = '}'; let d = 'enum';").is_empty());
  assert_eq!(
    shapes("enum E { A(char) } // '\nenum F { B { c: u8 } }"),
    vec![
      ("A".to_string(), Shape::Newtype),
      ("B".to_string(), Shape::Struct)
    ]
  );

  // `enum` must be a whole token, and a nested enum inside a function body is
  // still a declaration the rule covers.
  assert!(variants_in("struct Renumber { enumerate: u8 }").is_empty());
  assert_eq!(
    shapes("fn f() { enum E { A { x: u8 } } }"),
    vec![("A".to_string(), Shape::Struct)]
  );
}
