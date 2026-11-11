// swift-tools-version: 6.0
//
// R2, the Swift probe for coremlit's EmbeddingGemma 2 kit. Not part of the
// Rust build: it is driven only by `swift run -c release eg2probe <out>` (see
// README.md), which dumps what FluidUse -- the Swift implementation of
// EmbeddingGemma 2 -- feeds Core ML and gets back, over coremlit's committed
// corpus, into `../fixtures/goldens/swift_probe.json`.
//
// FluidUse is pinned by EXACT revision, and `Package.resolved` pins its own
// dependency (FluidAudio) as well. FluidUse's input builders are `private`, so
// the probe cannot call them; it hooks Core ML's predict methods instead and
// hashes the tensors FluidUse hands over (see `Hook.swift`). Nothing here
// modifies FluidUse, and nothing here writes into the model directory.

import PackageDescription

let package = Package(
  name: "eg2probe",
  platforms: [.macOS(.v15)],
  dependencies: [
    .package(
      url: "https://github.com/FluidInference/FluidUse.git",
      revision: "b51f7adfbade197f0a9c802e34ba8a98cd4ecbc9")
  ],
  targets: [
    .executableTarget(
      name: "eg2probe",
      dependencies: [
        .product(name: "FluidUse", package: "FluidUse")
      ],
      // The probe swizzles Objective-C methods through process-wide mutable
      // state; Swift 5 mode keeps that explicit and simple instead of
      // annotating every global for strict concurrency.
      swiftSettings: [.swiftLanguageMode(.v5)]
    )
  ]
)
