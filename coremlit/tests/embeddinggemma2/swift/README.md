# R2: the Swift probe for the EmbeddingGemma 2 kit

R2 is a reference dump made by FluidUse, the Swift implementation of
EmbeddingGemma 2, at revision `b51f7adfbade197f0a9c802e34ba8a98cd4ecbc9`
(Apache-2.0), over coremlit's committed corpus
(`../fixtures/goldens/{text,audio,image}.json`). The dump is
`../fixtures/goldens/swift_probe.json`. This package only produces it;
comparing it with the Rust kit is the Rust tests' job.

## What it records

For every item, in order:

- `ids`, text items only: the token ids from a public `EmbeddingGemma2Tokenizer`
  built from the model directory's `tokenizer.json`, using the same
  `encode(_:maxLength: 512)` the manager uses. FluidUse's tokenizer splits no
  special literal; the dump records what it does.
- `captures`: one entry per Core ML prediction FluidUse made for the item, in
  call order (audio window: the audio model, then the text function; image:
  the vision function, then the text function). Each has the loaded model's
  `configuration.functionName` (`null` is the default function) and, for every
  `MLMultiArray` input, `{name, shape, dtype, sha256}`. The digest is SHA-256
  over the elements in logical row-major order, little-endian, with strides
  honoured (padding never reaches it).
- `vector`: FluidUse's output vector, f32 as the shortest decimal that
  round-trips.

The corpus is every text item whose `recognize.ids` has at most 512 entries
(input: `prefix + text`, `prompt: .none`), every audio clip window (the clip is
cut to `cut.first_samples` first; each `(start, samples)` window is embedded on
its own), and one synthetic image per budget (70: 336x432, 140: 480x624, 280:
672x912) built from `_corpus.pattern_pixel`.

FluidUse's input builders are `private`, so the probe does not call them.
`Hook.swift` swizzles `MLModel`'s four public predict selectors
(`predictionFromFeatures:` with `completionHandler:` / `options:completionHandler:`
/ `error:` / `options:error:`) on `MLModel` and every Core ML subclass that
defines them. None of the four is reached: FluidUse's
`try await model.prediction(from:)` is CoreML's async overload, which wraps the
input in an `MLGenericPredictionRequest` and calls
`-[MLModel submitPredictionRequest:completionHandler:]` on the loaded
`MLDelegateModel`. The probe hooks that selector too and reads the request's
`inputFeatures`; `provenance.selector_calls` counts the calls per selector.
A prediction reached through more than one hooked selector is recorded once.

`provenance` also carries the FluidUse revision, `swift --version`,
`sw_vers`, the CPU, the model directory, the compute units of each model (the
FluidUse defaults, and the units observed on the loaded models), and
`coregraphics_redraw`: for each budget and each candidate source colour space,
how many R/G/B values FluidUse's CoreGraphics redraw changes at the image's own
size. The probe feeds FluidUse an image in the first colour space for which that
is zero.

## Regenerating

```sh
cd coremlit/tests/embeddinggemma2/swift
EMBEDDINGGEMMA2_TEST_MODELS=/Volumes/rollroll/models/embeddinggemma-2-coreml \
  swift run -c release eg2probe ../fixtures/goldens/swift_probe.json
```

The first build fetches FluidUse and FluidAudio from GitHub; `Package.resolved`
pins both. Needs macOS 15 (multifunction Core ML models).

The probe never writes into the model directory. It loads with
`EmbeddingGemma2Manager.load(from:)` and, because `EmbeddingGemma2Audio.load`
and `EmbeddingGemma2Vision.load` take no directory, sets
`EMBEDDINGGEMMA2_MODEL_DIR` in-process so neither can reach the ModelStore
download. FluidUse compiles the `.mlpackage`s with `MLModel.compileModel(at:)`
into Core ML's temporary directory (`TMPDIR`), not into the model directory.

## Host dependence

The output vectors depend on the host: the text model runs on the Neural
Engine, whose fp16 results differ between chips and OS builds. The input
digests do not: they hash tensors FluidUse builds on the CPU from the corpus,
so any host reproduces them bit for bit. Compare vectors by cosine with a
tolerance, and inputs by digest.
