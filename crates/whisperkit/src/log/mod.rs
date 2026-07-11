//! Leveled logging with a caller-installable sink.
//!
//! Ports `ArgmaxCore/Logging.swift`: a level gate (`LogLevel`), a callback
//! that REPLACES default output when installed (Swift parity — the callback
//! is a redirect, not a tee), and the resident-memory probe backing
//! `Logging.logCurrentMemoryUsage`. The optional `tracing` feature
//! additionally mirrors every emitted message as a `tracing` event.

use core::fmt;
use std::sync::Mutex;

#[cfg(test)]
mod tests;

/// Verbosity gate, ordered `Debug < Info < Error < None`.
///
/// Ports `Logging.LogLevel` (`ArgmaxCore/Logging.swift`); a message is
/// emitted when its level is at or above the configured level, and `None`
/// silences everything.
#[derive(
  Debug,
  Clone,
  Copy,
  PartialEq,
  Eq,
  PartialOrd,
  Ord,
  Hash,
  derive_more::Display,
  derive_more::IsVariant,
)]
#[display("{}", self.as_str())]
#[non_exhaustive]
pub enum LogLevel {
  /// Everything, including per-step diagnostics.
  Debug,
  /// Progress and lifecycle messages.
  Info,
  /// Failures only.
  Error,
  /// Nothing at all.
  None,
}

impl LogLevel {
  /// Stable snake_case name of the level.
  #[inline(always)]
  pub const fn as_str(&self) -> &'static str {
    match self {
      Self::Debug => "debug",
      Self::Info => "info",
      Self::Error => "error",
      Self::None => "none",
    }
  }
}

/// Error parsing a [`LogLevel`] name.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("unknown log level name")]
pub struct ParseLogLevelError(());

impl core::str::FromStr for LogLevel {
  type Err = ParseLogLevelError;

  fn from_str(s: &str) -> Result<Self, Self::Err> {
    Ok(match s {
      "debug" => Self::Debug,
      "info" => Self::Info,
      "error" => Self::Error,
      "none" => Self::None,
      _ => return Err(ParseLogLevelError(())),
    })
  }
}

/// The message sink installed via [`Logger::set_callback`].
pub type LoggingCallback = Box<dyn Fn(LogLevel, &str) + Send + Sync>;

struct LoggerState {
  level: LogLevel,
  callback: Option<LoggingCallback>,
}

/// A leveled logger with an optional replacing callback.
///
/// Instantiable so tests own their instances; the pipeline shares one
/// process-wide instance internally. When a callback is installed it
/// REPLACES stderr output entirely, mirroring Swift's
/// `Logging.loggingCallback` semantics.
pub struct Logger {
  state: Mutex<LoggerState>,
}

impl fmt::Debug for Logger {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    let state = self.state.lock().expect("logger lock poisoned");
    f.debug_struct("Logger")
      .field("level", &state.level)
      .field("callback", &state.callback.as_ref().map(|_| "<installed>"))
      .finish()
  }
}

impl Logger {
  /// A logger gating at `level` with no callback installed.
  pub const fn new(level: LogLevel) -> Self {
    Self {
      state: Mutex::new(LoggerState {
        level,
        callback: None,
      }),
    }
  }

  /// Replaces the gate level.
  pub fn set_level(&self, level: LogLevel) {
    self.state.lock().expect("logger lock poisoned").level = level;
  }

  /// Installs the sink that replaces default stderr output.
  pub fn set_callback(&self, callback: LoggingCallback) {
    self.state.lock().expect("logger lock poisoned").callback = Some(callback);
  }

  /// Emits `args` at `level` if the gate allows it.
  ///
  /// `LogLevel::None` messages are never emitted regardless of the gate
  /// (there is no "log at none" concept; it exists only as a gate value).
  pub fn log(&self, level: LogLevel, args: fmt::Arguments<'_>) {
    if level.is_none() {
      return;
    }
    let state = self.state.lock().expect("logger lock poisoned");
    if state.level.is_none() || level < state.level {
      return;
    }
    let message = std::fmt::format(args);
    #[cfg(feature = "tracing")]
    match level {
      LogLevel::Debug => tracing::debug!("{message}"),
      LogLevel::Info => tracing::info!("{message}"),
      LogLevel::Error => tracing::error!("{message}"),
      LogLevel::None => {}
    }
    match state.callback.as_ref() {
      Some(callback) => callback(level, &message),
      _ => eprintln!("[whisperkit {level}] {message}"),
    }
  }
}

/// This process's resident memory footprint, in bytes.
///
/// Ports `Logging.getMemoryUsage` (`ArgmaxCore/Logging.swift`): a
/// `task_info(MACH_TASK_BASIC_INFO)` query on the current task. `None`
/// when the kernel call fails.
pub fn resident_memory_bytes() -> Option<u64> {
  // SAFETY: `mach_task_basic_info` is a plain-old-data C struct; the
  // all-zero bit pattern is a valid value for every field.
  let mut info: libc::mach_task_basic_info = unsafe { core::mem::zeroed() };
  let mut count = (core::mem::size_of::<libc::mach_task_basic_info>()
    / core::mem::size_of::<libc::natural_t>()) as libc::mach_msg_type_number_t;
  // SAFETY: `mach_task_self()` (mach2's maintained binding — libc deprecated
  // its own in mach2's favor) names the current task; `info` is a zeroed,
  // properly sized and aligned MACH_TASK_BASIC_INFO out-struct and `count`
  // carries its capacity in words per the documented in-out contract, so
  // the kernel writes only within `info`'s bounds. The result code is
  // checked before `info` is read.
  let result = unsafe {
    libc::task_info(
      mach2::traps::mach_task_self(),
      libc::MACH_TASK_BASIC_INFO,
      (&raw mut info).cast(),
      &mut count,
    )
  };
  (result == libc::KERN_SUCCESS).then_some(info.resident_size)
}
