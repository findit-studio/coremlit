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
//! `crates/coremlit/tests/support/model_gate_report.rs` for the mechanism.
//!
//! That module lives under `tests/` because every other caller is a test
//! binary, and this one `#[path]`-hops to it rather than keeping a second copy
//! — the `tests/support/coremlit_dir.rs` convention, one level further. The hop
//! is `#[cfg(test)]`, so a published crate never compiles it.

use std::path::PathBuf;

#[path = "../tests/support/model_gate_report.rs"]
mod model_gate_report;

/// `<workspace>/Models/<sub>` unless `var` overrides it — the fallback every
/// in-lib `models_dir()` in this crate resolves. `sub` is empty for whisper,
/// whose gates read the `Models/` root itself.
fn root(var: &str, sub: &str) -> PathBuf {
  std::env::var_os(var).map_or_else(
    || {
      let models = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .join("Models");
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
