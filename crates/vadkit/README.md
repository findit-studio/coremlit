# vadkit

**Silero VAD on CoreML** — the FluidInference unified 256 ms artifact run through
the [`coremlit`](../coremlit) runtime, with all voice-activity *detection* logic
single-homed in the published [`silero`](https://github.com/Findit-AI/silero)
crate behind a backend seam.

macOS / Apple Silicon only (built on `coremlit`). Part of the
[coremlit](../../README.md) workspace.

## What it is

`vadkit` is the CoreML **model layer** for Silero VAD plus the thin wiring that
lets it drive `silero`'s detector:

- **`VadModel`** — wraps the FluidInference unified Silero VAD graph: one 256 ms
  (4096-sample) chunk of 16 kHz mono audio in, one speech probability out, with
  the recurrent LSTM state and the 64-sample rolling context carried across
  chunks (the FluidAudio `VadManager` semantics). Typed errors, an exact I/O
  contract pinned at load, and a bit-exact Swift-trace oracle in the tests.
- **`CoreMlBackend`** — implements `silero::VadBackend` over `VadModel`
  (`frame_samples() == 4096`, `predict`/`reset` over the recurrent state).
- **`detect_speech` + re-exports** — `vadkit::detect_speech` forwards to
  `silero::detect_speech_with`; `SpeechSegmenter`, `detect_speech_with`,
  `SpeechOptions`, `SpeechSegment`, `SampleRate` are re-exported unchanged. So a
  consumer gets the full offline **and** streaming detection API with **zero**
  segmentation logic authored here.

### The `silero` relationship — one home for the logic

The thresholding, start/end hysteresis, `min_speech`/`min_silence`,
`speech_pad`, and force-splitting all live in the published `silero` crate and
stay there. `silero` owns a backend-agnostic detector behind a `VadBackend`
trait; its ONNX/ort backend is one implementation, and `vadkit`'s
`CoreMlBackend` is another. `vadkit` re-exports `silero`'s detector surface
wired to CoreML rather than re-implementing any of it — a `src/` grep gate
(`tests/reexport.rs`) pins that nothing here authors detection logic.

`vadkit` depends on `silero` with `default-features = false` (logic only), so
**`ort`/ONNX never enters `vadkit`'s — or a downstream `whisperkit`'s — runtime
graph** (`cargo tree -p vadkit -e no-dev -i ort` is empty). The ONNX stack
appears only as a dev-dependency, for the cross-backend characterization gate.

> The runtime `silero` dependency is pinned by git rev to the `0.5.0` seam until
> `silero 0.5.0` publishes to crates.io, after which it becomes a plain version
> dependency (no behavior change).

## Usage

```rust,no_run
use vadkit::{CoreMlBackend, SpeechOptions, detect_speech};

// Load the CoreML model as a silero backend.
let mut backend = CoreMlBackend::load(
    "Models/vadkit/silero-vad-unified-256ms-v6.2.1.mlmodelc",
)?;

// One-shot offline detection over a 16 kHz mono buffer.
let segments = detect_speech(&mut backend, &samples, SpeechOptions::default())?;
for seg in &segments {
    println!("speech {:.2}s..{:.2}s", seg.start_seconds(), seg.end_seconds());
}
# Ok::<(), silero::Error>(())
```

Streaming drives the re-exported `silero::SpeechSegmenter` over the backend's
per-frame `predict`. `whisperkit` consumes `vadkit` a different way — behind its
`vadkit` feature, `whisperkit::silero_vad::SileroVad` plugs the Silero model
into whisperkit's own frame-level VAD seam for long-form chunking.

## Model & geometry

Adopted from Hugging Face, revision-pinned, never republished (the
alignkit/speakerkit adopted-model precedent):

| | |
|---|---|
| Repo | [`FluidInference/silero-vad-coreml`](https://huggingface.co/FluidInference/silero-vad-coreml) |
| Revision | `b419383c55c110e2c9271fa6ee0ea83d03c70d96` |
| Artifact | `silero-vad-unified-256ms-v6.2.1.mlmodelc` (ships pre-compiled; no `.mlpackage`) |
| License | MIT |

The revision and per-file SHA-256 are pinned in `tests/model_io.rs`. The model
is **not** committed; `Models/vadkit/` is gitignored and fetched dev-time:

```text
hf download FluidInference/silero-vad-coreml \
  --include "silero-vad-unified-256ms-v6.2.1*" \
  --revision b419383c55c110e2c9271fa6ee0ea83d03c70d96 \
  --local-dir Models/vadkit
```

**I/O contract** (all f32, pinned exactly): `audio_input [1, 4160]` (64 context +
4096 new) → `vad_output [1, 1, 1]` (one probability, a noisy-OR of eight
sigmoids in `[0, 1]`); the recurrent LSTM state is explicit feature I/O —
`hidden_state`/`cell_state [1, 128]` → `new_hidden_state`/`new_cell_state
[1, 128]` (the artifact declares an empty `stateSchema`; it is not an `MLState`
model). At 16 kHz that is one probability per 256 ms — an 8× coarser frame than
`silero`'s ONNX geometry (512 samples), which `silero`'s
geometry-parameterized detector consumes unchanged.

## Compute placement — the honest statement

`vadkit` selects compute units like the sibling kits (`ComputeUnits::All` by
default, letting CoreML schedule across the Neural Engine / GPU / CPU). It does
**not** claim the model "runs on the ANE": `coremlit` has no `MLComputePlan`
introspection to prove per-layer residency, and this graph's tail is
LSTM-dominated, which CoreML places on CPU. What the model gates *do* pin is a
compute-placement characterization — every `ComputeUnits` selection produces
**bit-identical** output on the fixture audio (measured worst |Δ| = 0), so the
default schedule is safe. The STFT front end is convolution-shaped and
ANE/GPU-eligible where CoreML chooses to place it; the crate states measured
behavior rather than marketing placement.

## Oracles & gates

- **Swift-trace parity** (`tests/parity_swift.rs`) — committed per-chunk
  probability traces from the real FluidAudio `VadManager`; `vadkit` reproduces
  them bit-for-bit (worst |Δ| = 0 across 217 chunks).
- **I/O + state + context** (`tests/model_io.rs`, `tests/model_state.rs`) —
  exact shape/dtype/SHA pins, state round-trip, and a misaligned-context guard
  (a one-sample context skew must change the probability).
- **No-duplication + re-export** (`tests/reexport.rs`) — the `src/` grep gate,
  `silero`'s detector scenarios replayed over a CoreML-shaped mock backend, and
  a model-gated end-to-end `detect_speech` pinned two-sided.
- **Cross-backend characterization** (`tests/cross_backend.rs`) — segment-level
  agreement against the `silero` ONNX stack (different model version *and*
  geometry, so behavioral agreement, not bit parity), two mutations proven red.
- **fp16 sweep** — the artifact joins `coremlit`'s MIL fp16-guard audit as a
  clean-control vendor (all STFT `sqrt` sites guarded at `2⁻²⁴`).

Model-gated tests are `#[ignore]` and run against a local `Models/vadkit`.

## Licensing

MIT end to end — see [`NOTICE`](NOTICE) for the two model attributions (upstream
Silero VAD, and FluidInference's CoreML conversion). The `vadkit` Rust source is
licensed under the workspace terms (MIT OR Apache-2.0).
