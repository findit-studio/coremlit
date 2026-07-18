// swift-tools-version: 5.10
//
// The argmax Swift ORACLE for `tests/parity_argmax_swift.rs` (design spec
// §5.1). Not part of the Rust build: it is driven only by
// `regen_goldens.sh`, which runs its single XCTest to dump argmax's own
// reading of its own decoded tensors into `../fixtures/golden_argmax_swift/`.
//
// It is an OUT-OF-TREE package that depends on the `argmax-oss-swift`
// checkout by PATH, deliberately: argmax's `SpeakerSegmenterModel.init`,
// `SpeakerEmbedderModel.embed` and `SpeakerEmbedding` are all `internal`, so
// the only way to run their pipeline and see its tensors is `@testable
// import SpeakerKit` from a TEST target (SwiftPM builds path dependencies
// with `-enable-testing` in debug). Nothing here writes to that checkout —
// no file of it is modified, and its build products land in this package's
// own gitignored `.build/`.
//
// `ARGMAX_SWIFT_SRC` overrides the checkout location; the default is the
// sibling of the coremlit workspace (`…/findit-studio/argmax-oss-swift`),
// which is where it lives on the machine this oracle was generated on.

import Foundation
import PackageDescription

let argmaxSource = Context.environment["ARGMAX_SWIFT_SRC"] ?? "../../../../../argmax-oss-swift"

let package = Package(
  name: "argmax-tensor-dump",
  platforms: [.macOS(.v13)],
  dependencies: [
    .package(path: argmaxSource)
  ],
  targets: [
    .testTarget(
      name: "ArgmaxTensorDump",
      dependencies: [
        .product(name: "SpeakerKit", package: "argmax-oss-swift"),
        .product(name: "WhisperKit", package: "argmax-oss-swift"),
      ]
    )
  ]
)
