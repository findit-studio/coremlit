//! SRT/VTT/JSON transcript writers. Ports `ResultWriter.swift` in full: the
//! `ResultWriting` protocol (`:6-10`) and its `formatTime`/`formatSegment`/
//! `formatTiming` default methods (`:12-38`), and the three conformers
//! `WriteJSON` (`:40-67`), `WriteSRT` (`:69-102`), `WriteVTT` (`:104-134`).
//!
//! Swift's `write(result:to:options:)` third parameter, `options: [String:
//! Any]?`, is dropped everywhere: every conformer's body ignores it
//! (`:53`, `:76`, `:111` all declare it only to never read it), so no Rust
//! signature threads an equivalent. Swift's per-conformer `Result<String,
//! Error>` return (the written file's `URL.absoluteString`) becomes
//! `Result<PathBuf, WriteError>` here: a typed error in place of the boxed
//! existential, and the actual written path rather than a stringly-typed
//! URL.
//!
//! `formatTime`/`formatSegment`/`formatTiming` are default methods on
//! Swift's protocol, callable on any conformer; this port has no per-writer
//! state they could close over, so all three become free functions.
//! [`format_time`] is the only one of the three this task's own interface
//! calls out as `pub` (mirroring `ResultWriting`'s own public method
//! surface, `ResultWriter.swift:9`); `format_segment`/`format_timing` stay
//! private, consumed only by [`srt_content`]/[`vtt_content`] below.

use std::path::{Path, PathBuf};

use crate::result::TranscriptionResult;

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------
// format_time / format_segment / format_timing
// ---------------------------------------------------------------------

/// Formats `seconds` as a timestamp, truncating (never rounding) to
/// millisecond precision. Ports `ResultWriting.formatTime`
/// (`ResultWriter.swift:14-25`): `hrs`/`mins`/`secs` come from truncating
/// division and remainder (Swift's `Int(_:)` cast on a `Float` truncates
/// toward zero, same as Rust's `as i32`; `seconds` is always non-negative
/// in this domain, so truncation and `floor` agree); `msec` is `((seconds
/// - seconds.floor()) * 1000.0) as i32` -- truncation, not rounding,
/// mirroring Swift's own `Int(...)` cast exactly (no `.rounded()` call in
/// the source). Renders `HH:MM:SS{marker}mmm` when `always_include_hours`
/// is `true` or the computed `hrs` is nonzero, else `MM:SS{marker}mmm`.
pub fn format_time(seconds: f32, always_include_hours: bool, decimal_marker: char) -> String {
  let hrs = (seconds / 3600.0) as i32;
  let mins = ((seconds % 3600.0) / 60.0) as i32;
  let secs = (seconds % 60.0) as i32;
  let msec = ((seconds - seconds.floor()) * 1000.0) as i32;

  if always_include_hours || hrs > 0 {
    format!("{hrs:02}:{mins:02}:{secs:02}{decimal_marker}{msec:03}")
  } else {
    format!("{mins:02}:{secs:02}{decimal_marker}{msec:03}")
  }
}

/// One SRT block: a 1-based index line, an always-hours `,`-marker timing
/// line, `text`, and a trailing blank line. Ports `ResultWriting.formatSegment`
/// (`ResultWriter.swift:27-31`).
fn format_segment(index: usize, start: f32, end: f32, text: &str) -> String {
  format!(
    "{index}\n{} --> {}\n{text}\n\n",
    format_time(start, true, ','),
    format_time(end, true, ','),
  )
}

/// One VTT block: an hours-when-nonzero `.`-marker timing line, `text`,
/// and a trailing blank line -- no index. Ports `ResultWriting.formatTiming`
/// (`ResultWriter.swift:33-37`).
fn format_timing(start: f32, end: f32, text: &str) -> String {
  format!(
    "{} --> {}\n{text}\n\n",
    format_time(start, false, '.'),
    format_time(end, false, '.'),
  )
}

// ---------------------------------------------------------------------
// srt_content / vtt_content / json_content
// ---------------------------------------------------------------------

/// Renders `result` as SRT subtitle content. Ports `WriteSRT.write`'s body
/// (`ResultWriter.swift:76-101`): a single running 1-based index across
/// the whole result; each segment contributes one `format_segment` block
/// per word when [`TranscriptionSegment::words_slice`](crate::result::TranscriptionSegment::words_slice)
/// is non-empty (`:83-88`), else one block for the segment itself
/// (`:89-93`).
pub fn srt_content(result: &TranscriptionResult) -> String {
  let mut content = String::new();
  let mut index = 1usize;
  for segment in result.segments_slice() {
    let words = segment.words_slice();
    if words.is_empty() {
      content.push_str(&format_segment(
        index,
        segment.start(),
        segment.end(),
        segment.text(),
      ));
      index += 1;
    } else {
      for word in words {
        content.push_str(&format_segment(
          index,
          word.start(),
          word.end(),
          word.word(),
        ));
        index += 1;
      }
    }
  }
  content
}

/// Renders `result` as WebVTT content. Ports `WriteVTT.write`'s body
/// (`ResultWriter.swift:111-133`): a `"WEBVTT\n\n"` header, then one
/// `format_timing` block per word when
/// [`TranscriptionSegment::words_slice`](crate::result::TranscriptionSegment::words_slice)
/// is non-empty, else one block for the segment -- the same per-segment
/// word/segment choice as [`srt_content`], minus the running index.
pub fn vtt_content(result: &TranscriptionResult) -> String {
  let mut content = String::from("WEBVTT\n\n");
  for segment in result.segments_slice() {
    let words = segment.words_slice();
    if words.is_empty() {
      content.push_str(&format_timing(
        segment.start(),
        segment.end(),
        segment.text(),
      ));
    } else {
      for word in words {
        content.push_str(&format_timing(word.start(), word.end(), word.word()));
      }
    }
  }
  content
}

/// Renders `result` as pretty-printed JSON. Ports `WriteJSON.write`
/// (`ResultWriter.swift:53-66`: a `JSONEncoder` with `.prettyPrinted`
/// output formatting).
///
/// **Documented deviation:** the emitted field names are this crate's own
/// serde derive (Rust `snake_case` field names, e.g. `avg_logprob` --
/// never Swift `Codable`'s default camelCase `avgLogprob`). Producing
/// Swift-shape JSON is a non-goal: this writer serializes *our*
/// [`TranscriptionResult`], not a wire-compatible mirror of Swift's
/// `Codable` output.
///
/// # Errors
/// [`WriteError::Serialize`] if `serde_json` fails to render `result`.
#[cfg(feature = "serde")]
pub fn json_content(result: &TranscriptionResult) -> Result<String, WriteError> {
  Ok(serde_json::to_string_pretty(result)?)
}

// ---------------------------------------------------------------------
// WriteError
// ---------------------------------------------------------------------

/// Failure writing a rendered transcript to disk. Swift's `ResultWriting`
/// protocol has no equivalent typed error -- every conformer folds every
/// failure into `Result<String, Error>`'s boxed existential `Error`
/// (`ResultWriter.swift:53,76,111`); this port narrows that to the
/// concrete causes a [`ResultWriter`] can actually hit.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum WriteError {
  /// Writing the rendered content to `path` failed.
  #[error("failed to write result file `{path}`: {source}", path = path.display())]
  Write {
    /// The path that failed to write.
    path: PathBuf,
    /// The underlying I/O error.
    source: std::io::Error,
  },
  /// Serializing the result to JSON failed.
  #[cfg(feature = "serde")]
  #[error("failed to serialize result: {0}")]
  Serialize(#[from] serde_json::Error),
}

// ---------------------------------------------------------------------
// ResultWriter
// ---------------------------------------------------------------------

/// Surface of Swift's `ResultWriting` protocol that survives the port
/// (`ResultWriter.swift:6-10`): the writer's output directory, and the
/// write operation. `formatTime` -- the protocol's third requirement --
/// becomes the free function [`format_time`] instead, since no conformer
/// here holds state it would need to close over; `options: [String:
/// Any]?` is dropped (see this module's doc comment).
pub trait ResultWriter {
  /// Directory new result files are written into.
  fn output_dir(&self) -> &Path;

  /// Renders `result` and writes it to `{output_dir}/{file_stem}.{ext}`,
  /// returning the written path.
  ///
  /// # Errors
  /// [`WriteError::Write`] if the underlying file write fails.
  fn write(&self, result: &TranscriptionResult, file_stem: &str) -> Result<PathBuf, WriteError>;
}

// ---------------------------------------------------------------------
// SrtWriter / VttWriter / JsonWriter
// ---------------------------------------------------------------------

/// Writes [`TranscriptionResult`]s as `.srt` files. Ports `WriteSRT`
/// (`ResultWriter.swift:69-102`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SrtWriter {
  output_dir: PathBuf,
}

impl SrtWriter {
  /// Creates a writer that writes into `output_dir`.
  pub fn new(output_dir: impl Into<PathBuf>) -> Self {
    Self {
      output_dir: output_dir.into(),
    }
  }
}

/// Writes `contents` to `path` atomically: staged into a sibling
/// temporary file, then renamed over the destination (rename within one
/// directory is atomic on macOS/POSIX). Ports Swift's
/// `String.write(..., atomically: true)` — a failed write can therefore
/// never truncate or half-replace an existing file, and a concurrent
/// reader sees either the old content or the new, never a partial
/// (phase-gate finding: `std::fs::write` truncates first).
fn write_atomic(path: &Path, contents: &str) -> Result<(), WriteError> {
  use std::io::Write as _;
  let map = |source| WriteError::Write {
    path: path.to_path_buf(),
    source,
  };
  // A UNIQUE staging sibling, arbitrated by `create_new` (O_EXCL): a
  // deterministic `<dest>.tmp` let two concurrent writers share one
  // staging inode — one could keep writing it after the other renamed it
  // into place, exposing partial content and breaking the old-or-new
  // guarantee (phase-gate round-3 finding) — and silently destroyed any
  // pre-existing file of that name. Collisions (same pid retrying, or a
  // stale leftover) skip to the next suffix instead of clobbering.
  let mut attempt = 0u32;
  let (mut file, tmp) = loop {
    let mut name = path.as_os_str().to_owned();
    name.push(format!(".{}.{attempt}.tmp", std::process::id()));
    let tmp = PathBuf::from(name);
    match std::fs::OpenOptions::new()
      .write(true)
      .create_new(true)
      .open(&tmp)
    {
      Ok(file) => break (file, tmp),
      Err(source) if source.kind() == std::io::ErrorKind::AlreadyExists && attempt < 1024 => {
        attempt += 1;
      }
      Err(source) => return Err(map(source)),
    }
  };
  let written = file.write_all(contents.as_bytes());
  // The descriptor must close before the rename: holding it open across
  // the swap is exactly the exposed-partial-content shape being fixed.
  drop(file);
  let staged = written.and_then(|()| std::fs::rename(&tmp, path));
  staged.map_err(|source| {
    // Best-effort cleanup; the original error is the one worth reporting.
    let _ = std::fs::remove_file(&tmp);
    map(source)
  })
}

impl ResultWriter for SrtWriter {
  fn output_dir(&self) -> &Path {
    &self.output_dir
  }

  fn write(&self, result: &TranscriptionResult, file_stem: &str) -> Result<PathBuf, WriteError> {
    let path = self.output_dir.join(format!("{file_stem}.srt"));
    write_atomic(&path, &srt_content(result))?;
    Ok(path)
  }
}

/// Writes [`TranscriptionResult`]s as `.vtt` files. Ports `WriteVTT`
/// (`ResultWriter.swift:104-134`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VttWriter {
  output_dir: PathBuf,
}

impl VttWriter {
  /// Creates a writer that writes into `output_dir`.
  pub fn new(output_dir: impl Into<PathBuf>) -> Self {
    Self {
      output_dir: output_dir.into(),
    }
  }
}

impl ResultWriter for VttWriter {
  fn output_dir(&self) -> &Path {
    &self.output_dir
  }

  fn write(&self, result: &TranscriptionResult, file_stem: &str) -> Result<PathBuf, WriteError> {
    let path = self.output_dir.join(format!("{file_stem}.vtt"));
    write_atomic(&path, &vtt_content(result))?;
    Ok(path)
  }
}

/// Writes [`TranscriptionResult`]s as pretty-printed `.json` files. Ports
/// `WriteJSON` (`ResultWriter.swift:40-67`).
#[cfg(feature = "serde")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JsonWriter {
  output_dir: PathBuf,
}

#[cfg(feature = "serde")]
impl JsonWriter {
  /// Creates a writer that writes into `output_dir`.
  pub fn new(output_dir: impl Into<PathBuf>) -> Self {
    Self {
      output_dir: output_dir.into(),
    }
  }
}

#[cfg(feature = "serde")]
impl ResultWriter for JsonWriter {
  fn output_dir(&self) -> &Path {
    &self.output_dir
  }

  fn write(&self, result: &TranscriptionResult, file_stem: &str) -> Result<PathBuf, WriteError> {
    let content = json_content(result)?;
    let path = self.output_dir.join(format!("{file_stem}.json"));
    write_atomic(&path, &content)?;
    Ok(path)
  }
}
