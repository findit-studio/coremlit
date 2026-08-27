//! Makes a modelless run's SKIPPED model gates say so, per test binary (#61).
//!
//! Every model gate in this repository is `#[ignore]`d — the convention — so a
//! run without `Models/` reports them as `ignored` and nothing else. That reads
//! identically whether the gates were deliberately skipped or whether the whole
//! suite evaporated, which is the third gap in #61: the absence is silent.
//!
//! [`report`] is a plain (NOT `#[ignore]`d) test body that each kit's shared
//! `common/mod.rs` calls, so every binary that already writes `mod common;`
//! gets it — including `coremlit-parity`'s oracle binaries, which
//! `#[path]`-include the very same files. It prints one line naming how many of
//! the binary's tests are model gates that did not run, and whether the models
//! root they need is on disk.
//!
//! # The count is libtest's, not a guess
//!
//! The binary re-executes ITSELF with `--list --ignored` — libtest's
//! ignored-ONLY listing, the same `RunIgnored::Only` filter and the same
//! `NAME: test` line shape CI's anti-vacuum guards count. So the number is what
//! libtest would actually select, not a scan of `#[ignore]` attributes that a
//! `cfg` could make a lie. `--list` executes no test, so the child cannot
//! recurse. Deliberately NOT a `cargo` invocation: `cargo` cannot nest inside a
//! `cargo test` run without deadlocking on the target-dir lock, which is why
//! the two gate-inventory checks next door are shell scripts rather than tests.
//!
//! # Written to the real fd 2, on purpose
//!
//! libtest CAPTURES `println!`/`eprintln!` from a test that PASSES and discards
//! it, so a passing test that "reports" something reports it to nobody unless
//! the reader remembers `--nocapture` — the same silence this is meant to end.
//! The capture is installed on the Rust-level stdout/stderr handles (and is
//! inherited by spawned threads), so the report writes to the inherited stderr
//! descriptor instead, in one `write` so lines from sibling tests cannot
//! interleave.
//!
//! # What it refuses, and what it does not
//!
//! Delete, un-`#[ignore]`, or `cfg` away a kit's gates and its binaries print
//! `0 of N`, in every ordinary run, next to that binary's own result line —
//! the "visibly report zero" half of the same property CI's `--list --ignored`
//! counts enforce by failing. It does NOT refuse a RENAMED gate: a rename
//! leaves the count alone. CI catches that where it matters, because a rename
//! breaks the gate plan's own selector (and, for CED, its `tiny::` filter).
//!
//! The one thing this DOES fail on is its own mechanism: a binary whose
//! unfiltered `--list` is empty could not have run this test, so an empty
//! listing means the self-exec is broken and every count it prints is a lie.
//!
//! The root probe is the DIRECTORY's existence, not an inventory of what is in
//! it: a half-staged tree — siglip's 33 MB tokenizer sidecar without its 748 MB
//! towers, say — reports `present`, and its gates then fail loudly on the
//! missing file. Naming the root is what turns "ignored" into "ignored, and
//! this is where the models would have been"; proving a tree COMPLETE is the
//! per-kit `model_io` gates' job, and CI's checksum step's.

use std::{
  io::Write,
  path::{Component, Path, PathBuf},
  process::Command,
};

/// Folds `a/../b` down to `b` so a root reads as a path rather than as the
/// expression that produced it. Purely lexical — the roots that most need
/// printing are the ones that do not exist, which `canonicalize` refuses.
fn tidy(path: &Path) -> PathBuf {
  let mut out = PathBuf::new();
  for component in path.components() {
    match component {
      Component::ParentDir
        if matches!(out.components().next_back(), Some(Component::Normal(_))) =>
      {
        out.pop();
      }
      other => out.push(other),
    }
  }
  out
}

/// Prints this binary's model-gate count and the state of the models roots
/// those gates need.
///
/// `roots` pairs each env var that overrides a root with the path this
/// binary's own resolver produced for it — pass `common::models_dir()` and
/// friends, never a second copy of the fallback, so the report cannot drift
/// from the gates it describes.
///
/// Panics only if the self-listing described in the module docs comes back
/// empty or fails to run.
pub fn report(roots: &[(&str, PathBuf)]) {
  let name = env!("CARGO_CRATE_NAME");
  let exe = std::env::current_exe()
    .unwrap_or_else(|e| panic!("model-gate report: cannot locate this test binary: {e}"));

  let list = |args: &[&str]| -> usize {
    let out = Command::new(&exe)
      .args(args)
      .output()
      .unwrap_or_else(|e| panic!("model-gate report: `{name} {}` failed: {e}", args.join(" ")));
    assert!(
      out.status.success(),
      "model-gate report: `{name} {}` exited {}",
      args.join(" "),
      out.status
    );
    String::from_utf8_lossy(&out.stdout)
      .lines()
      .filter(|line| line.ends_with(": test"))
      .count()
  };

  let total = list(&["--list"]);
  let gated = list(&["--list", "--ignored"]);
  assert!(
    total != 0,
    "model-gate report: `{name} --list` listed no tests, yet this test is running in it — \
     the self-listing is broken, so any count reported from it would be fiction"
  );

  let mut line =
    format!("model-gates | {name}: {gated} of {total} tests are #[ignore]d model gates");
  line.push_str(if gated == 0 {
    " (none — this binary gates on no models)"
  } else {
    " and did not run here"
  });
  let mut missing = false;
  for (var, root) in roots {
    let present = root.exists();
    missing |= !present;
    line.push_str(&format!(
      "; {var}={} {}",
      tidy(root).display(),
      if present { "present" } else { "MISSING" }
    ));
  }
  if gated > 0 {
    line.push_str(if missing {
      " -> stage the models (README, \"Getting models\") before running them"
    } else {
      " -> run them with `-- --ignored`"
    });
  }
  line.push('\n');

  // The inherited stderr descriptor, which libtest's capture does not touch.
  // `ManuallyDrop` because dropping a `File` closes its fd, and fd 2 belongs to
  // the process, not to this test. Best-effort: a report that cannot be written
  // must not turn a healthy suite red.
  //
  // SAFETY: fd 2 is open for the whole life of the process (libtest redirects
  // the Rust-level handles, never the descriptor), it is only ever written to
  // here, and `ManuallyDrop` keeps the `File` from closing it.
  let mut fd2 = std::mem::ManuallyDrop::new(unsafe {
    <std::fs::File as std::os::fd::FromRawFd>::from_raw_fd(2)
  });
  let _ = fd2.write_all(line.as_bytes());
}
