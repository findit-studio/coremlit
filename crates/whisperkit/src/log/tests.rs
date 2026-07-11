use std::sync::{Arc, Mutex};

use super::*;
use std::sync::Arc as StdArc;

#[test]
fn levels_gate_in_order() {
  assert!(LogLevel::Debug < LogLevel::Info);
  assert!(LogLevel::Info < LogLevel::Error);
  assert!(LogLevel::Error < LogLevel::None);
  assert_eq!(LogLevel::Info.as_str(), "info");
  assert_eq!(LogLevel::None.to_string(), "none");
  assert_eq!("error".parse::<LogLevel>().unwrap(), LogLevel::Error);
  assert!("verbose".parse::<LogLevel>().is_err());
}

#[test]
fn callback_receives_gated_messages() {
  let seen: Arc<Mutex<Vec<(LogLevel, String)>>> = Arc::default();
  let sink = Arc::clone(&seen);
  let logger = Logger::new(LogLevel::Info);
  logger.set_callback(StdArc::new(move |level, msg| {
    sink.lock().unwrap().push((level, msg.to_string()));
  }));
  logger.log(LogLevel::Debug, format_args!("hidden"));
  logger.log(LogLevel::Error, format_args!("shown {}", 42));
  let seen = seen.lock().unwrap();
  assert_eq!(
    seen.as_slice(),
    &[(LogLevel::Error, "shown 42".to_string())]
  );
}

#[test]
fn level_none_silences_everything() {
  let seen: Arc<Mutex<Vec<(LogLevel, String)>>> = Arc::default();
  let sink = Arc::clone(&seen);
  let logger = Logger::new(LogLevel::None);
  logger.set_callback(StdArc::new(move |level, msg| {
    sink.lock().unwrap().push((level, msg.to_string()));
  }));
  logger.log(LogLevel::Error, format_args!("dropped"));
  assert!(seen.lock().unwrap().is_empty());
}

#[test]
fn set_level_regates_at_runtime() {
  let seen: Arc<Mutex<Vec<String>>> = Arc::default();
  let sink = Arc::clone(&seen);
  let logger = Logger::new(LogLevel::Error);
  logger.set_callback(StdArc::new(move |_, msg| {
    sink.lock().unwrap().push(msg.to_string())
  }));
  logger.log(LogLevel::Info, format_args!("early"));
  logger.set_level(LogLevel::Debug);
  logger.log(LogLevel::Info, format_args!("late"));
  assert_eq!(seen.lock().unwrap().as_slice(), &["late".to_string()]);
}

#[test]
fn resident_memory_is_plausible_and_repeatable() {
  let first = resident_memory_bytes().expect("running process resides in memory");
  assert!(
    first > 1_000_000,
    "resident {first} bytes implausibly small"
  );
  assert!(resident_memory_bytes().is_some());
}

#[test]
fn reentrant_callback_does_not_deadlock() {
  let logger = StdArc::new(Logger::new(LogLevel::Debug));
  let seen: StdArc<Mutex<Vec<String>>> = StdArc::default();
  let sink = StdArc::clone(&seen);
  let inner = StdArc::clone(&logger);
  logger.set_callback(StdArc::new(move |level, msg: &str| {
    sink.lock().unwrap().push(msg.to_string());
    if !msg.starts_with("re:") {
      // Reconfigure AND re-log from inside the callback — must not deadlock.
      inner.set_level(LogLevel::Info);
      inner.log(level, format_args!("re: {msg}"));
    }
  }));
  logger.log(LogLevel::Error, format_args!("outer"));
  assert_eq!(
    seen.lock().unwrap().as_slice(),
    &["outer".to_string(), "re: outer".to_string()]
  );
}

#[test]
fn clear_callback_restores_default_routing() {
  let seen: StdArc<Mutex<Vec<String>>> = StdArc::default();
  let sink = StdArc::clone(&seen);
  let logger = Logger::new(LogLevel::Debug);
  logger.set_callback(StdArc::new(move |_, msg: &str| {
    sink.lock().unwrap().push(msg.to_string());
  }));
  logger.log(LogLevel::Info, format_args!("captured"));
  logger.clear_callback();
  logger.log(LogLevel::Info, format_args!("to stderr, not the sink"));
  assert_eq!(seen.lock().unwrap().as_slice(), &["captured".to_string()]);
}
