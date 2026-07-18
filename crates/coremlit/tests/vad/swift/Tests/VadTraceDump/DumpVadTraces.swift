//  The FluidAudio Swift reference dumper — the ORACLE behind
//  `crates/coremlit/tests/vad/parity_swift.rs` (design spec §6 model-layer gate).
//
//  It runs FluidAudio's OWN `VadManager`, not a reimplementation of it:
//
//      VadManager.process(_ samples: [Float]) -> [VadResult]
//
//  which internally splits the audio into 4096-sample chunks, prepends the
//  previous chunk's 64-sample tail as context, repeat-last-pads a short final
//  chunk, and carries the LSTM hidden/cell state across chunks
//  (`FluidAudio/Sources/FluidAudio/VAD/VadManager.swift`). The per-chunk
//  `VadResult.probability` those produce IS the quantity `vadkit::VadModel`
//  must reproduce — so if vadkit's port of that stitching is wrong (a skewed
//  context, a zero-padded final chunk, a dropped state field, a wrong first-
//  chunk zero-context), the committed trace diverges and the Rust gate fails.
//
//  # Compute placement is MATCHED, and pinned to `.cpuOnly`
//
//  The model is loaded ONCE here on `.cpuOnly` and handed to `VadManager` via
//  its `init(config:vadModel:)` initializer, so the config's `computeUnits`
//  never re-loads it — the whole trace runs on the CPU, deterministically. The
//  golden records `"cpu_only"`, and `parity_swift.rs` asserts its own
//  `ComputeUnits::CpuOnly` matches. Same reason the golden carries the audio's
//  FNV-1a: an input mismatch must fail as a HARNESS bug, never as a fidelity
//  number (the alignkit/speakerkit Gate-1 lesson).
//
//  # Regeneration
//
//      crates/coremlit/tests/vad/swift/regen_goldens.sh
//
//  See that script for the env contract. This test is a generator, not an
//  assertion suite; it is only ever run by that script.

import CoreML
import Foundation
import FluidAudio
import XCTest

// MARK: - Golden schema (mirrored by `parity_swift.rs`)

/// One 256 ms chunk: its index, the unpadded sample count fed for it (the
/// final chunk is shorter and repeat-last-padded inside `VadManager`), and the
/// speech probability `VadManager` produced.
private struct GoldenChunk: Encodable {
  let chunkIndex: Int
  let unpaddedSamples: Int
  let probability: Float
}

private struct Golden: Encodable {
  let fixture: String
  let generator: String
  let fluidAudioRevision: String
  /// Placement of the VAD model, as `coremlit::ComputeUnits` spells it. The
  /// Rust gate asserts its own placement matches.
  let computeUnits: String
  let sampleRate: Int
  let chunkSize: Int
  let contextSize: Int
  let stateSize: Int
  let modelInputSize: Int
  let inputSamples: Int
  /// FNV-1a-64 over the LE bytes of the samples fed to `VadManager` — the
  /// input-match proof.
  let inputFnv1a: String
  let chunkCount: Int
  /// Re-running the whole clip reproduced every probability bit-for-bit (only
  /// measured on the first fixture; `nil` elsewhere).
  let determinismVerified: Bool?
  let chunks: [GoldenChunk]
}

// MARK: - Dumper

final class DumpVadTraces: XCTestCase {
  /// Writes one golden JSON per fixture in `VADKIT_FIXTURES`.
  func testDumpGoldens() async throws {
    let modelURL = URL(fileURLWithPath: try env("VADKIT_MODEL"))
    let outDir = URL(fileURLWithPath: try env("VADKIT_GOLDEN_OUT"))
    let revision = ProcessInfo.processInfo.environment["FLUIDAUDIO_REVISION"] ?? "unknown"
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

    // "name=/abs/path.wav;name=/abs/path.wav"
    let fixtures: [(name: String, path: String)] = try env("VADKIT_FIXTURES")
      .split(separator: ";")
      .map { entry in
        let parts = entry.split(separator: "=", maxSplits: 1)
        guard parts.count == 2 else {
          throw DumpError.badEnvironment("VADKIT_FIXTURES entry '\(entry)' is not name=path")
        }
        return (String(parts[0]), String(parts[1]))
      }
    XCTAssertFalse(fixtures.isEmpty, "VADKIT_FIXTURES is empty")

    guard FileManager.default.fileExists(atPath: modelURL.path) else {
      throw DumpError.badEnvironment("missing VAD model artifact: \(modelURL.path)")
    }

    // Each `VadManager` gets a FRESHLY loaded `.cpuOnly` model, created and
    // transferred to the actor in one step so Swift 6 region-based isolation
    // can send the non-Sendable `MLModel` in with no shared local use (the
    // placement is `.cpuOnly` either way — this is what the golden records and
    // the Rust gate pins against). Loading is cheap; the model is tiny.
    func makeManager() throws -> VadManager {
      let configuration = MLModelConfiguration()
      configuration.computeUnits = .cpuOnly
      let model = try MLModel(contentsOf: modelURL, configuration: configuration)
      return VadManager(config: VadConfig(computeUnits: .cpuOnly), vadModel: model)
    }

    for (index, fixture) in fixtures.enumerated() {
      let samples = try readPcm16Mono16k(URL(fileURLWithPath: fixture.path))
      XCTAssertFalse(samples.isEmpty, "\(fixture.name): decoded zero samples")

      let manager = try makeManager()
      let results = try await manager.process(samples)

      // `VadManager` strides by chunkSize; the last chunk may be short.
      let chunkSize = VadManager.chunkSize
      let expectedChunks = (samples.count + chunkSize - 1) / chunkSize
      XCTAssertEqual(
        results.count, expectedChunks,
        "\(fixture.name): VadManager returned \(results.count) chunks, expected \(expectedChunks)")
      XCTAssertGreaterThan(results.count, 0, "\(fixture.name): zero chunks — nothing to compare")

      // The golden is only worth committing if it is reproducible. Prove it on
      // the first fixture by re-running the whole clip and demanding
      // bit-identical probabilities.
      var determinism: Bool?
      if index == 0 {
        let managerAgain = try makeManager()
        let again = try await managerAgain.process(samples)
        let identical =
          again.count == results.count
          && zip(results, again).allSatisfy { $0.probability == $1.probability }
        XCTAssertTrue(identical, "\(fixture.name): VadManager is not reproducible")
        determinism = identical
      }

      let chunks: [GoldenChunk] = results.enumerated().map { (c, result) in
        let start = c * chunkSize
        let unpadded = min(chunkSize, samples.count - start)
        return GoldenChunk(
          chunkIndex: c, unpaddedSamples: unpadded, probability: result.probability)
      }

      let golden = Golden(
        fixture: fixture.name,
        generator: "crates/coremlit/tests/vad/swift/Tests/VadTraceDump/DumpVadTraces.swift",
        fluidAudioRevision: revision,
        computeUnits: "cpu_only",
        sampleRate: VadManager.sampleRate,
        chunkSize: chunkSize,
        contextSize: VadState.contextLength,
        stateSize: 128,
        modelInputSize: chunkSize + VadState.contextLength,
        inputSamples: samples.count,
        inputFnv1a: fnv1aHex(samples),
        chunkCount: results.count,
        determinismVerified: determinism,
        chunks: chunks
      )

      let encoder = JSONEncoder()
      encoder.outputFormatting = [.sortedKeys]
      let path = outDir.appendingPathComponent("\(fixture.name).json")
      try encoder.encode(golden).write(to: path)
      print("[dump] \(fixture.name): \(samples.count) samples, \(results.count) chunks -> \(path.path)")
    }
  }
}

// MARK: - Audio

/// Reads a 16 kHz mono 16-bit PCM WAV as `[Float]`, scaling by `1/32768`.
///
/// Deliberately NOT `AudioConverter`: the gate needs both sides to feed the
/// model the SAME float array, and the Rust side reads the WAV with `hound`
/// (`tests/common/mod.rs`'s `load_wav_16k_mono`). This is that function's exact
/// semantics, so the two agree by construction — and the FNV-1a in the golden
/// proves it rather than assuming it. Byte-for-byte the same parser as the
/// speakerkit oracle's `readPcm16Mono16k`.
private func readPcm16Mono16k(_ url: URL) throws -> [Float] {
  let bytes = try Data(contentsOf: url)
  func u16(_ at: Int) -> Int { Int(bytes[at]) | Int(bytes[at + 1]) << 8 }
  func u32(_ at: Int) -> Int {
    Int(bytes[at]) | Int(bytes[at + 1]) << 8 | Int(bytes[at + 2]) << 16 | Int(bytes[at + 3]) << 24
  }
  guard bytes.count > 44, Array(bytes[0..<4]) == Array("RIFF".utf8),
    Array(bytes[8..<12]) == Array("WAVE".utf8)
  else {
    throw DumpError.badEnvironment("\(url.lastPathComponent): not a RIFF/WAVE file")
  }

  var offset = 12
  var data: Range<Int>?
  var channels = 0
  var rate = 0
  var bits = 0
  while offset + 8 <= bytes.count {
    let id = String(decoding: bytes[offset..<offset + 4], as: UTF8.self)
    let size = u32(offset + 4)
    let body = offset + 8
    if id == "fmt " {
      channels = u16(body + 2)
      rate = u32(body + 4)
      bits = u16(body + 14)
    } else if id == "data" {
      data = body..<min(body + size, bytes.count)
    }
    offset = body + size + (size % 2)  // RIFF chunks are word-aligned
  }
  guard let data else {
    throw DumpError.badEnvironment("\(url.lastPathComponent): no data chunk")
  }
  guard rate == 16000, channels == 1, bits == 16 else {
    throw DumpError.badEnvironment(
      "\(url.lastPathComponent): expected 16 kHz mono 16-bit, got \(rate) Hz / \(channels) ch / \(bits) bit"
    )
  }

  return stride(from: data.lowerBound, to: data.upperBound - 1, by: 2).map { at in
    Float(Int16(bitPattern: UInt16(bytes[at]) | UInt16(bytes[at + 1]) << 8)) / 32768.0
  }
}

/// FNV-1a-64 over the little-endian bytes of `samples` — byte-for-byte the
/// same construction as `crates/coremlit/tests/vad/common/mod.rs`'s `fnv1a_f32`.
private func fnv1aHex(_ samples: [Float]) -> String {
  var hash: UInt64 = 0xcbf2_9ce4_8422_2325
  for sample in samples {
    var bits = sample.bitPattern.littleEndian
    withUnsafeBytes(of: &bits) { raw in
      for byte in raw {
        hash ^= UInt64(byte)
        hash = hash &* 0x0000_0100_0000_01b3
      }
    }
  }
  return String(format: "%016lx", hash)
}

private enum DumpError: Error, CustomStringConvertible {
  case badEnvironment(String)

  var description: String {
    switch self {
    case .badEnvironment(let message): return message
    }
  }
}

private func env(_ name: String) throws -> String {
  guard let value = ProcessInfo.processInfo.environment[name], !value.isEmpty else {
    throw DumpError.badEnvironment("\(name) is not set (see regen_goldens.sh)")
  }
  return value
}
