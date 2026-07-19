//  The argmax Swift reference dumper — the ORACLE behind
//  `crates/coremlit/tests/speaker/parity_argmax_swift.rs` (design spec §5.1).
//
//  argmax ships a `DiarizeCLI`, but it emits ONLY final RTTM
//  (`Sources/ArgmaxCLI/DiarizeCLI.swift`: `SpeakerKit.generateRTTM(from:)`)
//  — i.e. the output of its VBx clustering, which `speakerkit` deliberately
//  does not own (spec §4, §7: clustering is `dia`'s). There is no flag,
//  anywhere in that CLI, that exposes an intermediate tensor. So the tier-1
//  fidelity surface — "did we read argmax's decoded tensors correctly" — has
//  no CLI oracle, and this dumper exists to be one.
//
//  It runs argmax's OWN pipeline objects, not a reimplementation of them:
//
//      SpeakerSegmenterModel.predict(audioArray:outputContinuation:)
//      SpeakerEmbedderModel.embed(segmenterOutput:)
//
//  and dumps the `[SpeakerEmbedding]` those produce. That array IS the thing
//  `speakerkit`'s `Extraction` carries, per (chunk, slot):
//
//      SpeakerEmbedding.windowIndex   -> Extraction chunk index `c`
//      SpeakerEmbedding.speakerIndex  -> Extraction slot index `s`
//      SpeakerEmbedding.activeFrames  -> segmentations[c][..][s]  (589 frames)
//      SpeakerEmbedding.embedding     -> raw_embeddings[c][s][..] (256 dims)
//
//  `windowIndex` deserves emphasis: argmax computes it as `chunkOffset(k) +
//  round(w * secondsPerStride)` = `k * 21 + w` (`SpeakerEmbedderModel.swift:
//  300`, `chunkStride = 21`), which is exactly the global chunk index
//  `source/argmax/mod.rs`'s `global_chunk(k, w)` derives. The Rust port's
//  central "grid theorem" is therefore checked against argmax's own
//  arithmetic rather than restated — and `bounded_windows`/`global_chunks`
//  below record argmax's own `bounded(windowIdx:)` verdict per chunk so the
//  Rust side can assert the surviving window set IS dia's chunk grid.
//
//  # Compute placement is MATCHED, and not by choice
//
//  `SpeakerPreEmbedderModel.init` HARDCODES `.cpuOnly`
//  (`SpeakerPreEmbedderModel.swift:14`) — argmax's fbank preprocessor cannot
//  run anywhere else. A `.all` Rust preprocessor against a `.cpuOnly` Swift
//  one would compare two different fbank computations and blame the
//  difference on the port. So this dumper pins ALL THREE models to
//  `.cpuOnly`, records that in the golden, and `parity_argmax_swift.rs`
//  asserts its own `ArgmaxComputeOptions` matches what the golden was
//  generated with. Same reason the golden carries the audio's FNV-1a: an
//  input mismatch must fail as a HARNESS bug, never as a fidelity number.
//
//  # Regeneration
//
//      crates/coremlit/tests/speaker/swift/regen_goldens.sh
//
//  See that script for the env contract. This test is a generator, not an
//  assertion suite; it is only ever run by that script.

import CoreML
import Foundation
import WhisperKit
import XCTest

@testable import SpeakerKit

// MARK: - Golden schema (mirrored by `parity_argmax_swift.rs`)

/// The host-class the golden was generated on — the four sysctl-derived fields
/// `coremlit`'s `HostClass` compares to gate the tight fidelity bounds. CoreML
/// `CpuOnly` floats are not contracted portable across macOS builds or chips
/// (#36), so a golden is trustworthy as a bit-exact oracle only on a matching
/// host. Mirrored by `parity_argmax_swift.rs`.
private struct GoldenHost: Encodable {
  let osBuild: String
  let osProductVersion: String
  let chip: String
  let arch: String
}

/// One consumed `(chunk, slot)`: exactly the pair of tensors `Extraction`
/// carries for it, as argmax's own Swift read them.
private struct GoldenSlot: Encodable {
  /// `SpeakerEmbedding.windowIndex` — argmax's `k * 21 + w`, which IS the
  /// `Extraction` chunk index.
  let chunk: Int
  /// `SpeakerEmbedding.speakerIndex` — the `Extraction` speaker slot.
  let slot: Int
  /// `SpeakerEmbedding.activeFrames`, the window's 589 `speaker_ids` values
  /// for this slot, as '0'/'1' (asserted binary below).
  let activeFrames: String
  /// `SpeakerEmbedding.nonOverlappedFrameRatio` — recorded so the Rust side
  /// can REPORT how often argmax's `minActiveRatio` clustering filter would
  /// have fired. Deliberately NOT ported (spec §5.2); it lives downstream of
  /// the embeddings, so it is out of this gate's surface.
  let nonOverlappedFrameRatio: Float
  /// `SpeakerEmbedding.embedding` — the raw 256-d WeSpeaker vector.
  let embedding: [Float]
}

/// Per argmax 30 s chunk `k`: its unpadded length and, crucially, which of
/// its 21 windows argmax's own `bounded(windowIdx:)` admits.
private struct GoldenChunk: Encodable {
  let chunkIndex: Int
  let unpaddedSamples: Int
  let waveformLengthSeconds: Float
  let windowsCount: Int
  /// The `w`s for which `SpeakerEmbedderContext.bounded(windowIdx:)` is true.
  let boundedWindows: [Int]
  /// `chunkOffset(k) + round(w * secondsPerStride)` for those `w` — argmax's
  /// own `windowIndex` formula (`SpeakerEmbedderModel.swift:300`), evaluated
  /// over EVERY bounded window regardless of speaker activity.
  ///
  /// Deliberately NOT read back from `run.embeddings[].windowIndex`, even
  /// though that field carries the identical formula's output
  /// (`GoldenSlot.chunk` below IS it): `SpeakerEmbedderModel.processChunk`
  /// only appends a `SpeakerEmbedding` for a `(window, speaker)` pair whose
  /// speaker is in `context.activeSpeakerIndices(for: windowIdx)`
  /// (`SpeakerEmbedderModel.swift:284-307` — the outer `for windowIdx` loop
  /// has no `else` branch, so a bounded window with ZERO active speakers
  /// contributes no `SpeakerEmbedding` and thus no `windowIndex` at all).
  /// Deriving `globalChunks` from emitted `windowIndex` values would
  /// therefore silently narrow it from "every bounded window" to "every
  /// bounded window with >=1 detected speaker" — a different, weaker
  /// invariant that happens to coincide with this formula on the three
  /// fixtures committed today (every bounded window here has >=1 consumed
  /// slot — verified against the goldens) but is not guaranteed to for a
  /// future fixture with a genuinely silent bounded window. `chunkOffset(for:)`
  /// IS read from argmax (`SpeakerEmbedderContext`, called below); only the
  /// stride arithmetic is restated, because argmax exposes no callable
  /// narrower than the private `processChunk` that returns this value.
  let globalChunks: [Int]
}

private struct Golden: Encodable {
  let fixture: String
  let generator: String
  let argmaxSwiftRevision: String
  /// The host-class this golden was generated on. `parity_argmax_swift.rs`'s
  /// host-aware gate enforces the exact/1e-6 fidelity bounds only when the
  /// running host matches this, and points a mismatch at regeneration.
  let generationHost: GoldenHost
  /// argmax's baseline tier: `W32A32` segmenter + `W16A16` embedder pair —
  /// `ArgmaxVariant::Baseline`.
  let variant: String
  /// Placement of each of the three models, as `coremlit::ComputeUnits`
  /// spells it. The Rust gate asserts its own options match these.
  let computeUnits: [String: String]
  let sampleRate: Int
  /// Sample count fed to `SpeakerSegmenterModel.predict`.
  let inputSamples: Int
  /// FNV-1a-64 over the LE bytes of those samples — the input-match proof.
  let inputFnv1a: String
  /// Diagnostic: what WhisperKit's own `AudioProcessor` decode of the same
  /// WAV yields. Not the gate's input (both sides are fed the array hashed
  /// above); recorded to show whether argmax's loader would have agreed.
  let whisperkitLoaderSamples: Int
  let whisperkitLoaderFnv1a: String
  let windowsCount: Int
  let framesPerWindow: Int
  let speakersCount: Int
  let embeddingDim: Int
  let chunkStrideSeconds: Int
  let secondsPerStride: Float
  /// Every `speaker_ids` value read was exactly 0.0 or 1.0 (the premise
  /// `activeFrames` is stored as a bit string, and that `onset` is inert).
  let activeFramesAreBinary: Bool
  /// Whether a FRESH `MLMultiArray(shape: [1, 64, 1767], .float16)` — the
  /// exact allocation `SpeakerEmbedderModel.processChunk` makes — comes back
  /// all-zero. argmax zero-fills only rows 0..<63 and leaves row 63
  /// uninitialized (`SpeakerEmbedderModel.swift:219-224`); the Rust port
  /// zeroes all 64. This records whether that divergence is even observable.
  let freshMaskAllocAllZero: Bool
  /// Re-running the whole pipeline on this fixture reproduced every dumped
  /// float bit-for-bit (only measured on the first fixture; `nil` elsewhere).
  let determinismVerified: Bool?
  let chunks: [GoldenChunk]
  let slots: [GoldenSlot]
}

// MARK: - Dumper

final class DumpArgmaxTensors: XCTestCase {
  /// Writes one golden JSON per fixture in `SPEAKERKIT_FIXTURES`.
  func testDumpGoldens() async throws {
    let modelsRoot = URL(fileURLWithPath: try env("ARGMAX_TEST_MODELS"))
    let outDir = URL(fileURLWithPath: try env("SPEAKERKIT_GOLDEN_OUT"))
    let revision = ProcessInfo.processInfo.environment["ARGMAX_SWIFT_REVISION"] ?? "unknown"
    try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

    // "name=/abs/path.wav;name=/abs/path.wav"
    let fixtures: [(name: String, path: String)] = try env("SPEAKERKIT_FIXTURES")
      .split(separator: ";")
      .map { entry in
        let parts = entry.split(separator: "=", maxSplits: 1)
        guard parts.count == 2 else {
          throw DumpError.badEnvironment("SPEAKERKIT_FIXTURES entry '\(entry)' is not name=path")
        }
        return (String(parts[0]), String(parts[1]))
      }
    XCTAssertFalse(fixtures.isEmpty, "SPEAKERKIT_FIXTURES is empty")

    let pipeline = try await Pipeline(modelsRoot: modelsRoot)

    // The host-class every golden in this run is stamped with. Read ONCE via
    // the identical sysctl keys `parity_argmax_swift.rs` reads, so string
    // equality is well-defined by construction; a read failure hard-fails the
    // dump rather than stamping a fake host.
    let generationHost = GoldenHost(
      osBuild: try sysctlString("kern.osversion"),
      osProductVersion: try sysctlString("kern.osproductversion"),
      chip: try sysctlString("machdep.cpu.brand_string"),
      arch: processArch
    )

    for (index, fixture) in fixtures.enumerated() {
      let samples = try readPcm16Mono16k(URL(fileURLWithPath: fixture.path))
      let run = try await pipeline.run(samples: samples)

      // The golden is only worth committing if it is reproducible. Prove it
      // on the first fixture by running the whole pipeline again and
      // demanding bit-identical floats (cheap: one extra fixture's compute).
      var determinism: Bool?
      if index == 0 {
        let again = try await pipeline.run(samples: samples)
        let identical =
          again.embeddings.count == run.embeddings.count
          && zip(run.embeddings, again.embeddings).allSatisfy { a, b in
            a.windowIndex == b.windowIndex && a.speakerIndex == b.speakerIndex
              && a.embedding == b.embedding && a.activeFrames == b.activeFrames
          }
        XCTAssertTrue(identical, "\(fixture.name): argmax's own pipeline is not reproducible")
        determinism = identical
      }

      var binary = true
      var fullWindow = true
      let slots: [GoldenSlot] = run.embeddings.map { embedding in
        if embedding.activeFrames.count != run.framesPerWindow {
          fullWindow = false
        }
        for value in embedding.activeFrames where value != 0.0 && value != 1.0 {
          binary = false
        }
        return GoldenSlot(
          chunk: embedding.windowIndex,
          slot: embedding.speakerIndex,
          activeFrames: String(embedding.activeFrames.map { $0 == 0.0 ? "0" : "1" }),
          nonOverlappedFrameRatio: embedding.nonOverlappedFrameRatio,
          embedding: embedding.embedding
        )
      }
      // `speaker_ids` being HARD binary is load-bearing for the Rust port
      // (it is what makes `onset` inert and `activeFrames` a bit string). If
      // it ever stops being so, this dump must fail rather than round.
      XCTAssertTrue(binary, "\(fixture.name): speaker_ids is not binary — the port's premise is void")
      // Each consumed slot must serialize EXACTLY `framesPerWindow` frames, and
      // the dump must consume at least one slot over a positive window. The
      // Rust loader now hard-requires the per-slot length; enforce it at
      // GENERATION too so a future dump cannot be vacuous. Note the `binary`
      // flag above is on its own vacuously TRUE for an empty `activeFrames`
      // vector (its `for` loop never runs), which is exactly the hole the
      // length and non-empty checks below close.
      XCTAssertGreaterThan(
        run.framesPerWindow, 0,
        "\(fixture.name): framesPerWindow is \(run.framesPerWindow) — a non-positive window makes the per-slot length check vacuous")
      XCTAssertFalse(
        run.embeddings.isEmpty,
        "\(fixture.name): argmax consumed zero (chunk, slot) slots — a dump with no slots would compare nothing")
      XCTAssertTrue(
        fullWindow,
        "\(fixture.name): a consumed slot's activeFrames is not exactly framesPerWindow (\(run.framesPerWindow)) frames — the dump would let the Rust segmentation comparison run over a truncated surface while still reporting full coverage")

      let whisperkit = whisperkitLoaderProbe(fixture.path)
      let golden = Golden(
        fixture: fixture.name,
        generator: "crates/coremlit/tests/speaker/swift/Tests/ArgmaxTensorDump/DumpArgmaxTensors.swift",
        argmaxSwiftRevision: revision,
        generationHost: generationHost,
        variant: "baseline",
        computeUnits: [
          "segmenter": "cpu_only",
          "preprocessor": "cpu_only",
          "embedder": "cpu_only",
        ],
        sampleRate: Pipeline.sampleRate,
        inputSamples: samples.count,
        inputFnv1a: fnv1aHex(samples),
        whisperkitLoaderSamples: whisperkit.count,
        whisperkitLoaderFnv1a: whisperkit.hash,
        windowsCount: run.windowsCount,
        framesPerWindow: run.framesPerWindow,
        speakersCount: run.speakersCount,
        embeddingDim: run.embeddings.first?.embedding.count ?? 0,
        chunkStrideSeconds: run.chunkStrideSeconds,
        secondsPerStride: run.secondsPerStride,
        activeFramesAreBinary: binary,
        freshMaskAllocAllZero: try freshMaskAllocIsAllZero(),
        determinismVerified: determinism,
        chunks: run.chunks,
        slots: slots.sorted { ($0.chunk, $0.slot) < ($1.chunk, $1.slot) }
      )

      let encoder = JSONEncoder()
      encoder.outputFormatting = [.sortedKeys]
      let path = outDir.appendingPathComponent("\(fixture.name).json")
      try encoder.encode(golden).write(to: path)
      print(
        "[dump] \(fixture.name): \(samples.count) samples, \(run.chunks.count) argmax chunk(s), "
          + "\(slots.count) consumed (chunk, slot) pairs, "
          + "host \(generationHost.osProductVersion)/\(generationHost.osBuild)/"
          + "\(generationHost.chip)/\(generationHost.arch) -> \(path.path)")
    }
  }
}

// MARK: - argmax's own pipeline

/// One run of argmax's segmenter + embedder over one fixture.
private struct PipelineRun {
  let embeddings: [SpeakerEmbedding]
  let chunks: [GoldenChunk]
  let windowsCount: Int
  let framesPerWindow: Int
  let speakersCount: Int
  let chunkStrideSeconds: Int
  let secondsPerStride: Float
}

/// argmax's three models, loaded from a local `speakerkit-coreml` root.
private struct Pipeline {
  static let sampleRate = 16000

  let segmenter: SpeakerSegmenterModel
  let embedder: SpeakerEmbedderModel

  init(modelsRoot: URL) async throws {
    let segmenterURL =
      modelsRoot
      .appendingPathComponent("speaker_segmenter/pyannote-v3/W32A32/SpeakerSegmenter.mlmodelc")
    let embedderDir = modelsRoot.appendingPathComponent("speaker_embedder/pyannote-v3/W16A16")
    for url in [
      segmenterURL,
      embedderDir.appendingPathComponent("SpeakerEmbedder.mlmodelc"),
      embedderDir.appendingPathComponent("SpeakerEmbedderPreprocessor.mlmodelc"),
    ] where !FileManager.default.fileExists(atPath: url.path) {
      throw DumpError.badEnvironment("missing model artifact: \(url.path)")
    }

    // `.cpuOnly` everywhere — see the file header. The preprocessor is
    // `.cpuOnly` whatever we pass (SpeakerPreEmbedderModel.swift:14), so the
    // other two follow it rather than the reverse.
    self.segmenter = try await SpeakerSegmenterModel(
      modelURL: segmenterURL,
      sampleRate: Self.sampleRate,
      concurrentWorkers: 1,
      useFullRedundancy: true,
      computeUnits: .cpuOnly
    )
    self.embedder = SpeakerEmbedderModel(
      modelURL: embedderDir.appendingPathComponent("SpeakerEmbedder.mlmodelc"),
      preprocessorModelURL: embedderDir.appendingPathComponent(
        "SpeakerEmbedderPreprocessor.mlmodelc"),
      // No PLDA projector: `Extraction` carries RAW embeddings — dia applies
      // its own PLDA downstream — and `ArgmaxSource` loads no PLDA model.
      pldaModelURL: nil,
      computeUnits: .cpuOnly
    )
    try await segmenter.loadModel()
    try await embedder.loadModel()
  }

  /// Runs argmax's pipeline exactly as `PyannoteDiarizerActor.initialize`
  /// does — segmenter yields one `SpeakerSegmenterOutput` per 30 s chunk,
  /// each is handed to `embedder.embed` — minus the clustering, which is the
  /// part `speakerkit` does not own.
  ///
  /// `predict` runs its (single) worker to completion and `finish()`es the
  /// continuation; the stream buffers unboundedly, so draining afterwards is
  /// equivalent to interleaving and strictly more deterministic.
  func run(samples: [Float]) async throws -> PipelineRun {
    let (stream, continuation) = AsyncStream.makeStream(of: SpeakerSegmenterOutput.self)
    try await segmenter.predict(audioArray: samples, outputContinuation: continuation)

    var embeddings: [SpeakerEmbedding] = []
    var chunks: [GoldenChunk] = []
    var windowsCount = 0
    var framesPerWindow = 0
    var speakersCount = 0
    var chunkStrideSeconds = 0
    var secondsPerStride: Float = 0

    for await output in stream {
      embeddings.append(contentsOf: try await embedder.embed(segmenterOutput: output))

      // argmax's own geometry, read back through its own context type — the
      // `bounded` verdict and the `windowIndex` formula are argmax's, not a
      // restatement of them.
      let context = SpeakerEmbedderContext(
        speakerActivity: try multiArray(output, "speaker_activity"),
        speakerIds: try multiArray(output, "speaker_ids"),
        overlappedSpeakerActivity: try multiArray(output, "overlapped_speaker_activity"),
        windowsCount: output.windowsCount,
        chunkStride: output.chunkStride,
        secondsPerWindow: output.secondsPerWindow,
        waveformLength: output.waveformLength
      )
      let bounded = (0..<context.windowsCount).filter { context.bounded(windowIdx: $0) }
      chunks.append(
        GoldenChunk(
          chunkIndex: output.chunkIndex,
          unpaddedSamples: Int((output.waveformLength * Float(Self.sampleRate)).rounded()),
          waveformLengthSeconds: output.waveformLength,
          windowsCount: context.windowsCount,
          boundedWindows: bounded,
          globalChunks: bounded.map { w in
            context.chunkOffset(for: output.chunkIndex)
              + Int((Float(w) * context.secondsPerStride).rounded())
          }
        ))
      windowsCount = context.windowsCount
      framesPerWindow = context.framesPerWindowCount
      speakersCount = context.speakersCount
      chunkStrideSeconds = output.chunkStride
      secondsPerStride = context.secondsPerStride
    }

    return PipelineRun(
      embeddings: embeddings.sorted { ($0.windowIndex, $0.speakerIndex) < ($1.windowIndex, $1.speakerIndex) },
      chunks: chunks.sorted { $0.chunkIndex < $1.chunkIndex },
      windowsCount: windowsCount,
      framesPerWindow: framesPerWindow,
      speakersCount: speakersCount,
      chunkStrideSeconds: chunkStrideSeconds,
      secondsPerStride: secondsPerStride
    )
  }
}

private func multiArray(_ output: SpeakerSegmenterOutput, _ name: String) throws -> MLMultiArray {
  guard let value = output.featureValue(for: name)?.multiArrayValue else {
    throw DumpError.badEnvironment("segmenter output '\(name)' is missing")
  }
  return value
}

// MARK: - Host

/// Reads a sysctl string by name using the two-call protocol (size query, then
/// read), NUL-terminated by `String(cString:)` and whitespace-trimmed. The Rust
/// gate reads the IDENTICAL keys via `/usr/sbin/sysctl -n`, so the recorded
/// strings compare by construction — do NOT substitute
/// `ProcessInfo.operatingSystemVersion` (it formats "15.5.0" where
/// `kern.osproductversion` is "15.5"). Throws rather than stamping a fake host.
private func sysctlString(_ name: String) throws -> String {
  var size = 0
  guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else {
    throw DumpError.badEnvironment("sysctl \(name): size query failed")
  }
  var buffer = [CChar](repeating: 0, count: size)
  guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else {
    throw DumpError.badEnvironment("sysctl \(name): read failed")
  }
  let value = String(cString: buffer).trimmingCharacters(in: .whitespacesAndNewlines)
  guard !value.isEmpty else {
    throw DumpError.badEnvironment("sysctl \(name): empty value")
  }
  return value
}

/// The process architecture governing which CPU kernels run in-process, spelled
/// as `coremlit`'s `HostClass` normalizes it (`arm64`/`x86_64`). Compile-time
/// arch IS the process arch.
private var processArch: String {
  #if arch(arm64)
    return "arm64"
  #elseif arch(x86_64)
    return "x86_64"
  #else
    #error("unsupported host architecture for golden generation")
  #endif
}

// MARK: - Audio

/// Reads a 16 kHz mono 16-bit PCM WAV as `[Float]`, scaling by `1/32768`.
///
/// Deliberately NOT `AudioProcessor.loadAudioAsFloatArray`: the gate needs
/// both sides to feed the models the SAME float array, and the Rust side
/// reads the WAV with `hound` (`tests/common/mod.rs`'s `load_wav_16k_mono`).
/// This is that function's exact semantics, so the two agree by construction
/// — and the FNV-1a in the golden proves it rather than assuming it. What
/// AudioProcessor would have produced is recorded separately, as a
/// diagnostic (`whisperkitLoader*`).
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
/// same construction as `tests/common/mod.rs`'s `fnv1a_f32`.
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

/// What WhisperKit's own AVFoundation-backed loader — the one `DiarizeCLI`
/// calls — makes of the same WAV. Diagnostic only.
private func whisperkitLoaderProbe(_ path: String) -> (count: Int, hash: String) {
  guard let samples = try? AudioProcessor.loadAudioAsFloatArray(fromPath: path) else {
    return (-1, "")
  }
  return (samples.count, fnv1aHex(samples))
}

// MARK: - Probes

/// Whether a fresh `[1, 64, 1767]` float16 `MLMultiArray` — the very
/// allocation `SpeakerEmbedderModel.processChunk` makes before zeroing only
/// its first 63 rows — arrives all-zero.
///
/// This is the observability check for the one KNOWN divergence in the Rust
/// port's mask construction (it zeroes all 64 rows; argmax leaves row 63
/// uninitialized). A `true` here means argmax's row 63 is zero in fact, so
/// the divergence is not merely inconsequential-by-row-independence — it is
/// not even present.
private func freshMaskAllocIsAllZero() throws -> Bool {
  let array = try MLMultiArray(shape: [1, 64, 1767], dataType: .float16)
  let words = array.dataPointer.bindMemory(to: UInt16.self, capacity: array.count)
  for index in 0..<array.count where words[index] != 0 {
    return false
  }
  return true
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
