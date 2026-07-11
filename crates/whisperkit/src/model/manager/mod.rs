//! [`ModelManager`]: the coalesced load/prewarm/unload lifecycle
//! orchestrator for the three CoreML models (mel, encoder, decoder) a
//! Whisper pipeline needs. Reshapes the deferred `ModelManager.swift`
//! lifecycle (`ArgmaxCore/ModelManager.swift:37-212`) for synchronous,
//! single-owner use; the actual load sequencing — which files load in
//! which order, and the shared load/prewarm shape — ports WhisperKit's own
//! `loadModels(prewarmMode:)` (`WhisperKit/Core/WhisperKit.swift:354-442`,
//! prewarm mode `:382-427`, unload `:487-499`, `modelStateCallback`
//! `:14-17`).
//!
//! See [`ModelManager::ensure_loaded`]'s doc for how this port replaces
//! Swift's `LoadModelsCoordinator` actor (`ModelManager.swift:73-86`,
//! `:214-232`) with plain `&mut self` serialization.

use std::path::PathBuf;

use crate::{
  error::ModelError,
  model::{LocalModelLoader, ModelLoader, ModelState, StateCallback},
  options::ComputeOptions,
};

#[cfg(test)]
mod tests;

// ---------------------------------------------------------------------
// LoadedModels
// ---------------------------------------------------------------------

/// The three loaded CoreML models a Whisper pipeline drives per window: the
/// mel-spectrogram feature extractor, the audio encoder, and the text
/// decoder. [`ModelManager::ensure_loaded`]/[`ModelManager::into_loaded`]
/// produce this; `CoreMlBackend::from_loaded` (`backend::coreml`) consumes
/// it.
#[derive(Debug)]
pub struct LoadedModels {
  mel: coremlit::Model,
  encoder: coremlit::Model,
  decoder: coremlit::Model,
}

impl LoadedModels {
  /// Wraps three already-loaded models.
  pub fn new(mel: coremlit::Model, encoder: coremlit::Model, decoder: coremlit::Model) -> Self {
    Self {
      mel,
      encoder,
      decoder,
    }
  }

  /// The mel-spectrogram feature extractor.
  #[inline(always)]
  pub const fn mel(&self) -> &coremlit::Model {
    &self.mel
  }

  /// The audio encoder.
  #[inline(always)]
  pub const fn encoder(&self) -> &coremlit::Model {
    &self.encoder
  }

  /// The text decoder.
  #[inline(always)]
  pub const fn decoder(&self) -> &coremlit::Model {
    &self.decoder
  }

  /// Unwraps into owned `(mel, encoder, decoder)` — the positional shape
  /// `CoreMlBackend::new` (`backend::coreml`) takes.
  #[must_use]
  pub fn into_parts(self) -> (coremlit::Model, coremlit::Model, coremlit::Model) {
    (self.mel, self.encoder, self.decoder)
  }
}

// ---------------------------------------------------------------------
// ModelManager
// ---------------------------------------------------------------------

/// Coalesced load/prewarm/unload orchestrator for a Whisper pipeline's
/// three CoreML models, resolved from a local folder (see the module doc).
pub struct ModelManager {
  folder: PathBuf,
  compute: ComputeOptions,
  state: ModelState,
  callback: Option<StateCallback>,
  models: Option<LoadedModels>,
}

impl std::fmt::Debug for ModelManager {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("ModelManager")
      .field("folder", &self.folder)
      .field("compute", &self.compute)
      .field("state", &self.state)
      .field("callback", &self.callback.as_ref().map(|_| "<installed>"))
      .field("models", &self.models)
      .finish()
  }
}

impl ModelManager {
  /// Builds a manager over `folder`, starting in [`ModelState::Unloaded`]
  /// with no callback installed. Not `const`: owned-path construction
  /// takes `impl Into<PathBuf>` (`options::Options::new`'s same non-const
  /// reason).
  pub fn new(folder: impl Into<PathBuf>, compute: ComputeOptions) -> Self {
    Self {
      folder: folder.into(),
      compute,
      state: ModelState::Unloaded,
      callback: None,
      models: None,
    }
  }

  /// The manager's current lifecycle state.
  #[inline(always)]
  pub const fn state(&self) -> ModelState {
    self.state
  }

  /// Installs the callback fired on every state transition, replacing any
  /// previously installed one.
  pub fn set_state_callback(&mut self, callback: StateCallback) -> &mut Self {
    self.callback = Some(callback);
    self
  }

  /// Builder form of [`Self::set_state_callback`].
  #[must_use]
  pub fn with_state_callback(mut self, callback: StateCallback) -> Self {
    self.set_state_callback(callback);
    self
  }

  /// Sequentially loads, then immediately drops, each model once — mel,
  /// decoder, then encoder (`WhisperKit.swift:382-427`'s `prewarmMode`
  /// order) — forcing ANE specialization/compilation up front rather than
  /// at first real inference. Ports `coremlit::Model::prewarm`'s
  /// load-then-drop shape, run once per model in sequence rather than
  /// racing all three. Transitions [`ModelState::Prewarming`] →
  /// [`ModelState::Prewarmed`] on success, or back to
  /// [`ModelState::Unloaded`] on failure (`ModelManager.swift:142-152`'s
  /// same failure-recovery shape).
  ///
  /// Unlike [`Self::ensure_loaded`], this is not idempotent: every call
  /// re-resolves `folder` and re-runs all three prewarms, matching
  /// `WhisperKit.swift`'s own `loadModels(prewarmMode:)`, which has no
  /// already-prewarmed short-circuit of its own (unlike the generic
  /// `ModelManager.prewarmModels()`, which does — that extra state-gated
  /// skip belongs to an abstraction this port does not carry over, since
  /// nothing here calls `prewarm()` more than once per real use).
  ///
  /// # Errors
  /// Whatever [`LocalModelLoader::resolve`] or `coremlit::Model::prewarm`
  /// returns.
  pub fn prewarm(&mut self) -> Result<(), ModelError> {
    self.transition(ModelState::Prewarming);
    match self.resolve_and_prewarm() {
      Ok(()) => {
        self.transition(ModelState::Prewarmed);
        Ok(())
      }
      Err(err) => {
        self.transition(ModelState::Unloaded);
        Err(err)
      }
    }
  }

  fn resolve_and_prewarm(&self) -> Result<(), ModelError> {
    let resolved = LocalModelLoader::new().resolve(&self.folder)?;
    coremlit::Model::prewarm(resolved.mel_ref(), self.compute.mel())?;
    coremlit::Model::prewarm(resolved.decoder_ref(), self.compute.decoder())?;
    coremlit::Model::prewarm(resolved.encoder_ref(), self.compute.encoder())?;
    Ok(())
  }

  /// Ensures the three models are loaded, returning the cached
  /// [`LoadedModels`] — idempotent: a call while already
  /// [`ModelState::Loaded`] just returns the existing models with no
  /// re-resolution or re-load. Otherwise resolves `folder` through
  /// [`LocalModelLoader`] and loads mel, decoder, then encoder (the same
  /// order [`Self::prewarm`] uses), transitioning [`ModelState::Loading`]
  /// → [`ModelState::Loaded`] on success, or back to
  /// [`ModelState::Unloaded`] on failure (`ModelManager.swift:180-190`'s
  /// same failure-recovery shape).
  ///
  /// **Sync reshape of Swift's coalescing actor:** Swift's
  /// `ensureModelsLoaded()` coalesces concurrent async callers onto one
  /// in-flight `Task` through a private `LoadModelsCoordinator` actor
  /// (`ModelManager.swift:73-86`, `LoadModelsCoordinator` `:214-232`) — a
  /// second caller arriving while a load is in flight awaits the SAME task
  /// rather than starting a second one. This port has no concurrent
  /// callers to coalesce: this method takes `&mut self`, so the borrow
  /// checker already forbids a second call from starting while a first is
  /// in flight — that would require two live `&mut` borrows of the same
  /// `ModelManager`, which does not compile. What is left of the actor's
  /// behavior is exactly its OWN pre-coalescing idempotency checks (`guard
  /// !isLoaded else { return }` at `ModelManager.swift:76`, repeated as
  /// `guard !self.isLoaded else { return }` once inside the coordinator's
  /// escaping closure at `:78`) — a cached-return when already loaded —
  /// which is what the `state() == Loaded` check below implements
  /// directly. No coordinator/task machinery is needed to get the same "a
  /// second caller reuses the first caller's result" outcome once
  /// concurrent entry is impossible by construction.
  ///
  /// # Errors
  /// Whatever [`LocalModelLoader::resolve`] or `coremlit::Model::load`
  /// returns.
  pub fn ensure_loaded(&mut self) -> Result<&LoadedModels, ModelError> {
    if self.state != ModelState::Loaded {
      self.load_now()?;
    }
    Ok(
      self
        .models
        .as_ref()
        .expect("state Loaded is only ever set together with models, in load_now()"),
    )
  }

  fn load_now(&mut self) -> Result<(), ModelError> {
    self.transition(ModelState::Loading);
    match self.resolve_and_load() {
      Ok(models) => {
        self.models = Some(models);
        self.transition(ModelState::Loaded);
        Ok(())
      }
      Err(err) => {
        self.transition(ModelState::Unloaded);
        Err(err)
      }
    }
  }

  fn resolve_and_load(&self) -> Result<LoadedModels, ModelError> {
    let resolved = LocalModelLoader::new().resolve(&self.folder)?;
    let mel = coremlit::Model::load(resolved.mel_ref(), self.compute.mel())?;
    let decoder = coremlit::Model::load(resolved.decoder_ref(), self.compute.decoder())?;
    let encoder = coremlit::Model::load(resolved.encoder_ref(), self.compute.encoder())?;
    Ok(LoadedModels::new(mel, encoder, decoder))
  }

  /// Releases the loaded models, transitioning [`ModelState::Unloading`] →
  /// [`ModelState::Unloaded`] unconditionally (Swift
  /// `WhisperKit.unloadModels()`, `WhisperKit.swift:487-499`, which has no
  /// "already unloaded" guard of its own — unlike the generic
  /// `ModelManager.unloadModels()`'s `guard modelState == .loaded ||
  /// .prewarmed`). A no-op on the model data itself when nothing was
  /// loaded, since dropping `None` does nothing.
  pub fn unload(&mut self) {
    self.transition(ModelState::Unloading);
    self.models = None;
    self.transition(ModelState::Unloaded);
  }

  /// [`Self::ensure_loaded`], then hands off ownership of the resulting
  /// [`LoadedModels`] — the construction-time path a `WhisperKit::new`
  /// (a later task) uses when its load-at-construction option is set.
  ///
  /// # Errors
  /// As [`Self::ensure_loaded`].
  pub fn into_loaded(mut self) -> Result<LoadedModels, ModelError> {
    self.ensure_loaded()?;
    Ok(
      self
        .models
        .take()
        .expect("ensure_loaded() above returned Ok, so models is populated"),
    )
  }

  /// Moves to `new`, firing the installed callback (if any) with the prior
  /// state (Swift `modelStateCallback(oldValue, modelState)`,
  /// `WhisperKit.swift:14-17`). Every public transition in this manager
  /// goes through here, so callback ordering is uniform.
  fn transition(&mut self, new: ModelState) {
    let old = self.state;
    self.state = new;
    if let Some(callback) = &self.callback {
      callback(Some(old), new);
    }
  }
}
