# online_oracle — Swift reference for the online-clusterer parity gate

This SwiftPM executable is the oracle behind
`crates/coremlit/tests/speaker/parity_online_swift.rs`. It `import`s the **local
FluidAudio checkout** and drives its `SpeakerManager.assignSpeaker` directly on
the same deterministic synthetic LCG embedding sequence the Rust harness
regenerates, then dumps the per-step decision trace (kind / id / centroid) as
JSON. That JSON, committed as
`../../fixtures/golden_online_swift/trace.json`, lets the Rust gate replay
Swift's decisions with **no Swift toolchain and no models**.

It emits **both** input-attestation hashes with a Swift FNV-1a-64 that is
byte-identical to the Rust `common::fnv1a_f32`:

- `inputFnv1a` — over the 48×256 raw embeddings. Reproducing the committed value
  proves the LCG + FNV mirror is faithful.
- `durationsFnv1a` — over the `[Float]` speech-duration sequence. This is the
  cross-language attestation of the SECOND `assignSpeaker` input (the one that
  gates new-vs-dropped at `minSpeechDuration`); the Rust gate asserts its own
  regenerated durations hash to this Swift-emitted value, so a Swift-side
  duration divergence the embedding hash cannot see is caught (finding M3).

## FluidAudio path dependency

The package depends on FluidAudio by path — a sibling of the coremlit
workspace, `…/findit-studio/FluidAudio` — resolved read-only. Override the
location with `FLUIDAUDIO_SRC` if your checkout lives elsewhere:

```sh
FLUIDAUDIO_SRC=/abs/path/to/FluidAudio swift run online_oracle > /dev/null
```

## Regenerate the committed trace

From this directory:

```sh
swift run online_oracle > ../../fixtures/golden_online_swift/trace.json
```

stdout carries only the JSON (the `[.prettyPrinted, .sortedKeys]` encoding the
committed trace uses); the `inputFnv1a` / `durationsFnv1a` / count diagnostics
print to stderr. After regenerating, run the Rust gate to confirm faithfulness
(it re-derives the sequence, re-hashes both inputs against the trace, and
compares every per-step decision and centroid):

```sh
cargo test -p coremlit --test speaker_parity_online_swift --all-features
```

Only the package `Package.swift` + `Sources/` and the committed trace are
tracked; `.build/` and `Package.resolved` are ignored.
