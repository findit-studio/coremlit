use std::sync::{Arc, Mutex};

use super::*;
use crate::audio::whisper::{model::ModelState, options::ComputeOptions};

#[test]
fn missing_folder_fails_and_restores_unloaded() {
  let mut manager = ModelManager::new("/nonexistent/models", ComputeOptions::new());
  assert_eq!(manager.state(), ModelState::Unloaded);
  assert!(manager.ensure_loaded().is_err());
  assert_eq!(
    manager.state(),
    ModelState::Unloaded,
    "failed load leaves a sane state"
  );
}

#[test]
#[allow(clippy::type_complexity)] // (Option<ModelState>, ModelState) IS the recorded transition.
fn transitions_fire_callback_with_old_and_new() {
  let seen: Arc<Mutex<Vec<(Option<ModelState>, ModelState)>>> = Arc::default();
  let sink = Arc::clone(&seen);
  let mut manager = ModelManager::new("/nonexistent/models", ComputeOptions::new())
    .with_state_callback(Box::new(move |old, new| {
      sink.lock().unwrap().push((old, new))
    }));
  let _ = manager.ensure_loaded();
  let seen = seen.lock().unwrap();
  // Loading fired, then the failure transition back to Unloaded.
  assert_eq!(
    seen.first(),
    Some(&(Some(ModelState::Unloaded), ModelState::Loading))
  );
  assert_eq!(
    seen.last(),
    Some(&(Some(ModelState::Loading), ModelState::Unloaded))
  );
}

#[test]
#[allow(clippy::type_complexity)] // (Option<ModelState>, ModelState) IS the recorded transition.
fn unload_on_nothing_resident_is_silent() {
  // Regression (task-10 review, Important): ModelManager.swift:195's
  // guard — no spurious Unloading/Unloaded callback pair when nothing is
  // loaded or prewarmed.
  let seen: Arc<Mutex<Vec<(Option<ModelState>, ModelState)>>> = Arc::default();
  let sink = Arc::clone(&seen);
  let mut manager = ModelManager::new("/nonexistent/models", ComputeOptions::new())
    .with_state_callback(Box::new(move |old, new| {
      sink.lock().unwrap().push((old, new))
    }));
  manager.unload();
  assert!(seen.lock().unwrap().is_empty(), "no transitions fired");
  assert_eq!(manager.state(), ModelState::Unloaded);
}
