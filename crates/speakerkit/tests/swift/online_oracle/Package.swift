// swift-tools-version: 6.0
//
// The online-clusterer Swift ORACLE for `tests/parity_online_swift.rs`. Not
// part of the Rust build (cargo ignores Swift sources): it is driven only by
// hand / by the regen command in this directory's README, which runs the
// executable to dump the per-step SpeakerManager decision trace into
// `../fixtures/golden_online_swift/trace.json`.
//
// It depends on the LOCAL FluidAudio checkout BY PATH (a SIBLING of the
// coremlit workspace: `…/findit-studio/FluidAudio`), read-only — nothing here
// writes into that checkout, and build products land in this package's own
// gitignored `.build/`. `FLUIDAUDIO_SRC` overrides the checkout location; the
// default is the sibling layout this repo is developed in.
//
// swift-tools 6.0 + macOS v14 mirror FluidAudio's own Package manifest (a
// dependent must meet the dependency's minimum deployment target). The oracle's
// own code compiles in Swift-5 language mode (`swiftLanguageModes`) — it is a
// single-threaded synchronous dumper with no need for strict concurrency.

import Foundation
import PackageDescription

let fluidSource = Context.environment["FLUIDAUDIO_SRC"] ?? "../../../../../../FluidAudio"

let package = Package(
  name: "online_oracle",
  platforms: [.macOS(.v14)],
  dependencies: [
    .package(path: fluidSource)
  ],
  targets: [
    .executableTarget(
      name: "online_oracle",
      dependencies: [
        .product(name: "FluidAudio", package: "FluidAudio")
      ]
    )
  ],
  swiftLanguageModes: [.v5]
)
