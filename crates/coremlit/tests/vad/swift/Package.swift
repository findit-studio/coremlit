// swift-tools-version: 6.0
//
// The FluidAudio Swift ORACLE for `tests/parity_swift.rs` (design spec §6
// model-layer gate). Not part of the Rust build: it is driven only by
// `regen_goldens.sh`, which runs its single XCTest to dump FluidAudio's own
// `VadManager` per-chunk speech probabilities into `../fixtures/golden_swift/`.
//
// It runs FluidAudio's OWN pipeline — `VadManager.process([Float])`, which
// performs the 4096-sample chunking, the 64-sample context stitching, the
// repeat-last padding of the short final chunk, and the recurrent-state
// carry-forward that `vadkit::VadModel` ports — over a pre-loaded `.cpuOnly`
// copy of the SAME `Models/vadkit` artifact. Everything it uses is PUBLIC
// (`VadManager`, `VadConfig`, `VadResult`), so this is a plain `import
// FluidAudio` (no `@testable`); it lives in a TEST target only to match the
// speakerkit oracle's `swift test`-driven shape. Nothing here writes to the
// FluidAudio checkout — its build products land in this package's own
// gitignored `.build/`.
//
// `FLUIDAUDIO_SRC` overrides the checkout location; the default is the sibling
// of the coremlit workspace (`…/findit-studio/FluidAudio`), which is where it
// lives on the machine this oracle was generated on.

import Foundation
import PackageDescription

let fluidAudioSource = Context.environment["FLUIDAUDIO_SRC"] ?? "../../../../../../FluidAudio"

let package = Package(
  name: "vad-trace-dump",
  platforms: [.macOS(.v14)],
  dependencies: [
    .package(path: fluidAudioSource)
  ],
  targets: [
    .testTarget(
      name: "VadTraceDump",
      dependencies: [
        .product(name: "FluidAudio", package: "FluidAudio")
      ]
    )
  ]
)
