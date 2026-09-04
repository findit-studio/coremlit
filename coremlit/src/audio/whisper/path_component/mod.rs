//! One rule, asked by every door here that builds a path out of a caller's
//! string: is that string ONE plain path component?
//!
//! Two doors take a caller-controlled string, append a fixed suffix, and join
//! the result to a directory the caller configured earlier —
//! [`ResultWriter::write`](crate::audio::whisper::result::writer::ResultWriter::write)'s
//! `file_stem` under the writer's `output_dir` (#114), and
//! [`detect_model_url`](crate::audio::whisper::model::detect_model_url)'s
//! `name` under its `folder` (#120). A join is not a concatenation:
//! `dir.join("../x")` names a file in `dir`'s PARENT and `dir.join("/x")`
//! discards `dir` altogether, so a string carrying path syntax silently
//! redirects the write, or the read, to a directory other than the one the
//! caller configured.
//!
//! [`single_path_component`] refuses exactly the spellings that change WHICH
//! directory a join resolves into, and nothing else: no Unicode
//! normalisation, no length cap, no reserved-name list, no case folding.
//! Whether the surviving name is one this filesystem will actually accept —
//! too long, already taken, on a read-only mount — is the filesystem's
//! business, reported by the `std::fs` call that runs into it.
//!
//! `.` and `..` are refused as the components they are, even though both call
//! sites append a suffix before joining (`..` in fact reaches
//! `folder/...mlmodelc`, an ordinary file, and escapes nothing). The rule is a
//! property of the caller's STRING, not of one caller's suffix: a door that
//! ever joins the string bare must get the same answer from it.
//!
//! `\` is refused alongside `/` though macOS treats it as an ordinary
//! filename byte. A transcript stem and a model name both travel — into an
//! archive entry, onto an SMB share, to a Windows client — and are read as
//! two components wherever they land; neither of these two doors has a caller
//! that wants one, so refusing it costs nothing here.

#[cfg(test)]
mod tests;

/// Why a caller's string is not one plain path component.
///
/// Crate-private on purpose: each door maps this into ITS own public error
/// ([`WriteError::FileStem`](crate::audio::whisper::result::writer::WriteError::FileStem),
/// [`ModelError::ModelName`](crate::audio::whisper::error::ModelError::ModelName)),
/// which is what a caller matches on. [`Self::reason`] is the phrase those
/// errors render.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) enum PathComponentDefect {
  /// The string is empty, so it names nothing to join.
  Empty,
  /// The string holds `/` or `\`, so joining it descends or escapes.
  Separator,
  /// The string holds a NUL byte, which no filesystem path can carry.
  Nul,
  /// The string is `.`, which names a directory rather than an entry in one.
  CurrentDirectory,
  /// The string is `..`, which names the parent directory.
  ParentDirectory,
}

impl PathComponentDefect {
  /// The phrase a door's error renders after the offending string, as
  /// `` `{string}` {reason} ``.
  pub(crate) const fn reason(self) -> &'static str {
    match self {
      Self::Empty => "is empty",
      Self::Separator => "contains a path separator (`/` or `\\`)",
      Self::Nul => "contains a NUL byte",
      Self::CurrentDirectory => "is `.`, which names a directory rather than an entry in one",
      Self::ParentDirectory => "is `..`, which names the parent directory",
    }
  }
}

/// `s` unchanged when it is one plain path component; its defect otherwise.
///
/// Accepted iff `s` is non-empty, holds no `/`, no `\` and no NUL byte, and is
/// neither `.` nor `..`. That is the whole rule — see this module's doc for
/// why nothing else belongs in it.
///
/// # Errors
/// The [`PathComponentDefect`] the string tripped, whose
/// [`reason`](PathComponentDefect::reason) each caller renders into its own
/// error type.
pub(crate) fn single_path_component(s: &str) -> Result<&str, PathComponentDefect> {
  if s.is_empty() {
    return Err(PathComponentDefect::Empty);
  }
  // A BYTE scan is exactly a scan for these three characters: `/`, `\` and NUL
  // are ASCII, and no byte of a multi-byte UTF-8 sequence is ever below 0x80,
  // so no `char` boundary can hide one and none can be spelled accidentally by
  // a continuation byte.
  for byte in s.bytes() {
    match byte {
      b'/' | b'\\' => return Err(PathComponentDefect::Separator),
      0 => return Err(PathComponentDefect::Nul),
      _ => {}
    }
  }
  // After the scan, so `../x` reports the separator that actually escapes
  // rather than a `..` prefix that does not appear on its own.
  match s {
    "." => Err(PathComponentDefect::CurrentDirectory),
    ".." => Err(PathComponentDefect::ParentDirectory),
    _ => Ok(s),
  }
}
