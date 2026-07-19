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

use std::{
  path::PathBuf,
  time::{Duration, Instant},
};

use crate::audio::whisper::{
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
  mel: crate::Model,
  encoder: crate::Model,
  decoder: crate::Model,
}

impl LoadedModels {
  /// Wraps three already-loaded models.
  pub fn new(mel: crate::Model, encoder: crate::Model, decoder: crate::Model) -> Self {
    Self {
      mel,
      encoder,
      decoder,
    }
  }

  /// The mel-spectrogram feature extractor.
  #[inline(always)]
  pub const fn mel(&self) -> &crate::Model {
    &self.mel
  }

  /// The audio encoder.
  #[inline(always)]
  pub const fn encoder(&self) -> &crate::Model {
    &self.encoder
  }

  /// The text decoder.
  #[inline(always)]
  pub const fn decoder(&self) -> &crate::Model {
    &self.decoder
  }

  /// Unwraps into owned `(mel, encoder, decoder)` — the positional shape
  /// `CoreMlBackend::new` (`backend::coreml`) takes.
  #[must_use]
  pub fn into_parts(self) -> (crate::Model, crate::Model, crate::Model) {
    (self.mel, self.encoder, self.decoder)
  }
}

// ---------------------------------------------------------------------
// ModelLoadTimings
// ---------------------------------------------------------------------

/// Per-model load/specialization durations a [`ModelManager`] observes while
/// bringing the encoder and decoder up, handed off by
/// [`ModelManager::into_loaded`] for a `WhisperKit` to fold into its
/// per-run [`crate::audio::whisper::result::TranscriptionTimings`]. Mirrors
/// the split Swift's `WhisperKit.loadModels(prewarmMode:)` records into
/// `currentTimings` (`WhisperKit.swift:396-423`): the **prewarm** pass's
/// per-model load elapsed is the *specialization* time (the one-time
/// ANE/graph specialization the throwaway prewarm load pays up front), and
/// the **real** load pass's per-model elapsed is the *load* time. The two
/// `*_specialization` durations therefore stay [`Duration::ZERO`] whenever
/// no prewarm pass ran ([`ModelManager::prewarm`] was not called), exactly
/// as Swift leaves them unset without a `prewarmMode` load.
///
/// The mel feature-extractor's own load is not split out here (Swift records
/// no per-mel field either); it is captured only in the whole-pass totals a
/// `WhisperKit` measures around [`ModelManager::prewarm`]/
/// [`ModelManager::into_loaded`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModelLoadTimings {
  encoder_load: Duration,
  decoder_load: Duration,
  encoder_specialization: Duration,
  decoder_specialization: Duration,
}

impl ModelLoadTimings {
  /// Time the real (non-prewarm) load pass spent loading the audio encoder.
  #[inline(always)]
  pub const fn encoder_load(&self) -> Duration {
    self.encoder_load
  }

  /// Time the real (non-prewarm) load pass spent loading the text decoder.
  #[inline(always)]
  pub const fn decoder_load(&self) -> Duration {
    self.decoder_load
  }

  /// Time the prewarm pass spent loading (specializing) the audio encoder;
  /// [`Duration::ZERO`] when no prewarm pass ran.
  #[inline(always)]
  pub const fn encoder_specialization(&self) -> Duration {
    self.encoder_specialization
  }

  /// Time the prewarm pass spent loading (specializing) the text decoder;
  /// [`Duration::ZERO`] when no prewarm pass ran.
  #[inline(always)]
  pub const fn decoder_specialization(&self) -> Duration {
    self.decoder_specialization
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
  timings: ModelLoadTimings,
}

impl std::fmt::Debug for ModelManager {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    f.debug_struct("ModelManager")
      .field("folder", &self.folder)
      .field("compute", &self.compute)
      .field("state", &self.state)
      .field("callback", &self.callback.as_ref().map(|_| "<installed>"))
      .field("models", &self.models)
      .field("timings", &self.timings)
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
      timings: ModelLoadTimings::default(),
    }
  }

  /// The manager's current lifecycle state.
  #[inline(always)]
  pub const fn state(&self) -> ModelState {
    self.state
  }

  /// Installs the callback fired on every state transition, replacing any
  /// previously installed one.
  #[inline(always)]
  pub fn set_state_callback(&mut self, callback: StateCallback) -> &mut Self {
    self.callback = Some(callback);
    self
  }

  /// Builder form of [`Self::set_state_callback`].
  #[must_use]
  #[inline(always)]
  pub fn with_state_callback(mut self, callback: StateCallback) -> Self {
    self.set_state_callback(callback);
    self
  }

  /// Sequentially loads, then immediately drops, each model once — mel,
  /// decoder, then encoder (`WhisperKit.swift:382-427`'s `prewarmMode`
  /// order) — forcing ANE specialization/compilation up front rather than
  /// at first real inference. Ports `crate::Model::prewarm`'s
  /// load-then-drop shape, run once per model in sequence rather than
  /// racing all three. Transitions [`ModelState::Prewarming`] →
  /// [`ModelState::Prewarmed`] on success, or back to
  /// [`ModelState::Unloaded`] on failure (`ModelManager.swift:142-152`'s
  /// same failure-recovery shape).
  ///
  /// State-gated like the generic `ModelManager.prewarmModels()`
  /// (`ModelManager.swift:131-139`): already [`ModelState::Prewarmed`] is
  /// a silent skip, and any resident-model state is rejected — Swift only
  /// prewarns from `.downloaded`, whose local-folder equivalent here is
  /// [`ModelState::Unloaded`] (there is no download layer; a folder on
  /// disk is definitionally "downloaded"). Rejecting
  /// [`ModelState::Loaded`] keeps the resident triple and the lifecycle
  /// state consistent: prewarm-over-loaded would otherwise relabel
  /// resident models `Prewarmed`, double-load on the next
  /// [`Self::ensure_loaded`], and on failure strand them behind
  /// [`Self::unload`]'s state guard.
  ///
  /// # Errors
  /// [`ModelError::InvalidState`] when called while models are resident
  /// ([`ModelState::Loaded`]); whatever [`LocalModelLoader::resolve`] or
  /// `crate::Model::prewarm` returns otherwise.
  pub fn prewarm(&mut self) -> Result<(), ModelError> {
    match self.state {
      // ModelManager.swift:132-134 — already prewarmed: skip silently.
      ModelState::Prewarmed => return Ok(()),
      ModelState::Loaded => {
        return Err(ModelError::InvalidState {
          expected: "unloaded (local models prewarm before loading)",
          actual: self.state.as_str(),
        });
      }
      _ => {}
    }
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

  fn resolve_and_prewarm(&mut self) -> Result<(), ModelError> {
    let resolved = LocalModelLoader::new().resolve(&self.folder)?;
    crate::Model::prewarm(resolved.mel_ref(), self.compute.mel())?;
    // The prewarm-pass per-model load is Swift's `*SpecializationTime`
    // (`WhisperKit.swift:401-403,420-422`): the throwaway load-then-drop
    // pays the one-time ANE/graph specialization so the real load below is
    // fast. `resolved` owns its paths and `compute` is `Copy`, so neither
    // outlives its use — the `self.timings` writes borrow freely between loads.
    let decoder_start = Instant::now();
    crate::Model::prewarm(resolved.decoder_ref(), self.compute.decoder())?;
    self.timings.decoder_specialization = decoder_start.elapsed();
    let encoder_start = Instant::now();
    crate::Model::prewarm(resolved.encoder_ref(), self.compute.encoder())?;
    self.timings.encoder_specialization = encoder_start.elapsed();
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
  /// Whatever [`LocalModelLoader::resolve`] or `crate::Model::load`
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

  fn resolve_and_load(&mut self) -> Result<LoadedModels, ModelError> {
    let resolved = LocalModelLoader::new().resolve(&self.folder)?;
    let mel = crate::Model::load(resolved.mel_ref(), self.compute.mel())?;
    // The real (non-prewarm) per-model load is Swift's `*LoadTime`
    // (`WhisperKit.swift:404-406,423-425`). Timed individually so a
    // `WhisperKit` can populate `encoder_load_time`/`decoder_load_time`
    // separately from the whole-pass `model_loading` total it measures
    // around `into_loaded`.
    let decoder_start = Instant::now();
    let decoder = crate::Model::load(resolved.decoder_ref(), self.compute.decoder())?;
    self.timings.decoder_load = decoder_start.elapsed();
    let encoder_start = Instant::now();
    let encoder = crate::Model::load(resolved.encoder_ref(), self.compute.encoder())?;
    self.timings.encoder_load = encoder_start.elapsed();
    Ok(LoadedModels::new(mel, encoder, decoder))
  }

  /// Releases the loaded models, transitioning [`ModelState::Unloading`] →
  /// [`ModelState::Unloaded`]. A complete no-op — no transitions, no
  /// callbacks — unless something is resident to release
  /// (`ModelManager.unloadModels()`'s `guard modelState == .loaded ||
  /// .prewarmed`, `ModelManager.swift:194-201`; the guard keeps a
  /// callback-driven UI from seeing spurious `Unloading`/`Unloaded`
  /// pairs. `WhisperKit.swift:487-499` is a separate, unguarded
  /// implementation — lifecycle semantics here follow `ModelManager`,
  /// the load sequencing follows `WhisperKit`).
  pub fn unload(&mut self) {
    if !matches!(self.state, ModelState::Loaded | ModelState::Prewarmed) {
      return;
    }
    self.transition(ModelState::Unloading);
    self.models = None;
    self.transition(ModelState::Unloaded);
  }

  /// [`Self::ensure_loaded`], then hands off ownership of the resulting
  /// [`LoadedModels`] together with the [`ModelLoadTimings`] observed while
  /// bringing them up — the construction-time path `WhisperKit::new` uses
  /// when its load-at-construction option is set, so it can fold the
  /// per-model load/specialization splits into each run's timings. The
  /// returned timings carry the specialization durations from a prior
  /// [`Self::prewarm`] (all-zero if none ran) as well as this call's
  /// per-model load durations.
  ///
  /// # Errors
  /// As [`Self::ensure_loaded`].
  pub fn into_loaded(mut self) -> Result<(LoadedModels, ModelLoadTimings), ModelError> {
    self.ensure_loaded()?;
    let timings = self.timings;
    let models = self
      .models
      .take()
      .expect("ensure_loaded() above returned Ok, so models is populated");
    Ok((models, timings))
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
