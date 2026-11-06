// R2: what FluidUse (the Swift EmbeddingGemma 2) feeds Core ML, and gets back,
// over coremlit's committed corpus. See README.md.
//
//   EMBEDDINGGEMMA2_TEST_MODELS=<model dir> swift run -c release eg2probe <out.json>

import CoreGraphics
import CoreML
import CryptoKit
import FluidUse
import Foundation

let fluidUseRevision = "b51f7adfbade197f0a9c802e34ba8a98cd4ecbc9"

// MARK: - Arguments and paths

func fail(_ message: String) -> Never {
  FileHandle.standardError.write(Data("eg2probe: \(message)\n".utf8))
  exit(1)
}

let arguments = CommandLine.arguments
guard arguments.count == 2 else { fail("usage: eg2probe <out.json>") }
let outputPath = arguments[1]
guard let modelsPath = ProcessInfo.processInfo.environment["EMBEDDINGGEMMA2_TEST_MODELS"],
  !modelsPath.isEmpty
else { fail("set EMBEDDINGGEMMA2_TEST_MODELS to the model directory") }
let modelDirectory = URL(fileURLWithPath: modelsPath, isDirectory: true)

// `coremlit/tests`, found from this source file: Sources/eg2probe/main.swift
// -> eg2probe -> Sources -> swift -> embeddinggemma2 -> tests.
let testsDirectory: URL = {
  if let override = ProcessInfo.processInfo.environment["EG2PROBE_TESTS_DIR"], !override.isEmpty {
    return URL(fileURLWithPath: override, isDirectory: true)
  }
  var url = URL(fileURLWithPath: #filePath)
  for _ in 0..<5 { url.deleteLastPathComponent() }
  return url
}()
let goldensDirectory = testsDirectory.appendingPathComponent("embeddinggemma2/fixtures/goldens")

// MARK: - Helpers

func hex(_ bytes: some Sequence<UInt8>) -> String { bytes.map { String(format: "%02x", $0) }.joined() }

func sha256(of url: URL) throws -> String {
  hex(SHA256.hash(data: try Data(contentsOf: url)))
}

func run(_ command: String, _ arguments: [String]) -> String {
  let process = Process()
  process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
  process.arguments = [command] + arguments
  let pipe = Pipe()
  process.standardOutput = pipe
  process.standardError = pipe
  do { try process.run() } catch { return "unavailable: \(error)" }
  let data = pipe.fileHandleForReading.readDataToEndOfFile()
  process.waitUntilExit()
  return String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
}

func readJSON(_ name: String) throws -> [String: Any] {
  let data = try Data(contentsOf: goldensDirectory.appendingPathComponent(name))
  guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
    fail("\(name) is not a JSON object")
  }
  return object
}

/// 16 kHz mono s16le PCM as `s / 32768`, read straight from the RIFF chunks
/// (WAVE_FORMAT_EXTENSIBLE included). An independent check on what
/// `EmbeddingGemma2Audio.samples(contentsOf:)` decodes through AVFoundation.
func riffSamples(_ url: URL) -> [Float]? {
  let bytes = [UInt8](try! Data(contentsOf: url))
  func u16(_ at: Int) -> Int { Int(bytes[at]) | Int(bytes[at + 1]) << 8 }
  func u32(_ at: Int) -> Int { u16(at) | u16(at + 2) << 16 }
  guard bytes.count > 12, bytes[0..<4].elementsEqual("RIFF".utf8), bytes[8..<12].elementsEqual("WAVE".utf8)
  else { return nil }
  var offset = 12
  var format: (tag: Int, channels: Int, rate: Int, bits: Int)?
  while offset + 8 <= bytes.count {
    let id = String(decoding: bytes[offset..<offset + 4], as: UTF8.self)
    let size = u32(offset + 4)
    let body = offset + 8
    if id == "fmt " {
      var tag = u16(body)
      if tag == 0xFFFE && size >= 26 { tag = u16(body + 24) }
      format = (tag, u16(body + 2), u32(body + 4), u16(body + 14))
    } else if id == "data" {
      guard let format, format.tag == 1, format.channels == 1, format.rate == 16_000, format.bits == 16
      else { return nil }
      let end = min(body + size, bytes.count)
      return stride(from: body, to: end - 1, by: 2).map {
        Float(Int16(truncatingIfNeeded: u16($0))) / 32768
      }
    }
    offset = body + size + (size & 1)
  }
  return nil
}

func computeUnitsName(_ units: MLComputeUnits) -> String {
  switch units {
  case .cpuOnly: "cpuOnly"
  case .cpuAndGPU: "cpuAndGPU"
  case .all: "all"
  case .cpuAndNeuralEngine: "cpuAndNeuralEngine"
  @unknown default: "unknown(\(units.rawValue))"
  }
}

/// Runs `body` with the recorder cleared and returns what it captured.
func captured<T>(_ body: () async throws -> T) async rethrows -> (T, [Capture]) {
  Recorder.shared.reset()
  let value = try await body()
  return (value, Recorder.shared.take())
}

// MARK: - Load

// FluidUse's defaults, from its public `load` signatures: the text model on
// the Neural Engine, the audio and vision models on the GPU.
let textUnits = MLComputeUnits.cpuAndNeuralEngine
let audioUnits = MLComputeUnits.cpuAndGPU
let visionUnits = MLComputeUnits.cpuAndGPU

for file in [
  "EmbeddingGemma2Text.mlpackage", "EmbeddingGemma2Audio.mlpackage", "EmbeddingGemma2Vision.mlpackage",
  "embeddings.bf16", "position_embeddings.f16", "tokenizer.json",
] where !FileManager.default.fileExists(atPath: modelDirectory.appendingPathComponent(file).path) {
  fail("\(file) is missing from \(modelDirectory.path)")
}
// `EmbeddingGemma2Audio.load` and `EmbeddingGemma2Vision.load` take no
// directory: they read this variable, and only without it do they reach the
// ModelStore download. Setting it keeps both on the local directory.
setenv("EMBEDDINGGEMMA2_MODEL_DIR", modelDirectory.path, 1)

Hook.install()
let tokenizer = try EmbeddingGemma2Tokenizer(
  tokenizerJsonURL: modelDirectory.appendingPathComponent("tokenizer.json"))
let manager = try await EmbeddingGemma2Manager.load(from: modelDirectory)
let audio = try await EmbeddingGemma2Audio.load(text: manager)
let vision = try await EmbeddingGemma2Vision.load(text: manager)
// The loaded models may be a subclass that only now exists.
Hook.install()
FileHandle.standardError.write(Data("hooked: \(Hook.installed.joined(separator: ", "))\n".utf8))

var refused: [JSON] = []
func refuse(_ modality: String, _ id: String, _ error: Error) {
  let message = (error as? LocalizedError)?.errorDescription ?? "\(error)"
  FileHandle.standardError.write(Data("refused \(modality) \(id): \(message)\n".utf8))
  refused.append(
    .object([("modality", .string(modality)), ("id", .string(id)), ("message", .string(message))]))
}

// MARK: - Text

let textGolden = try readJSON("text.json")
var textItems: [JSON] = []
var textSkipped = 0
for case let item as [String: Any] in textGolden["items"] as! [Any] {
  let id = item["id"] as! String
  let recognize = item["recognize"] as! [String: Any]
  guard (recognize["ids"] as! [Any]).count <= 512 else {
    textSkipped += 1
    continue
  }
  // The joint string: the prefix and the text, embedded with no prompt.
  let joint = (item["prefix"] as! String) + (item["text"] as! String)
  let ids = tokenizer.encode(joint, maxLength: 512)
  do {
    let (vector, captures) = try await captured { try await manager.embed(joint, prompt: .none) }
    textItems.append(
      .object([
        ("id", .string(id)),
        ("ids", .array(ids.map { .int(Int($0)) })),
        ("captures", .array(captures.map(\.json))),
        ("vector", .vector(vector)),
      ]))
  } catch {
    refuse("text", id, error)
  }
}
print("text: \(textItems.count) embedded, \(textSkipped) over 512 tokens skipped")

// MARK: - Audio

let audioGolden = try readJSON("audio.json")
var audioItems: [JSON] = []
for case let clip as [String: Any] in audioGolden["clips"] as! [Any] {
  let id = clip["id"] as! String
  let url = testsDirectory.appendingPathComponent(clip["file"] as! String)
  do {
    let fileSha = try sha256(of: url)
    var samples = try EmbeddingGemma2Audio.samples(contentsOf: url)
    let riff = riffSamples(url)
    let decoderMatchesRiff = riff.map { $0 == samples }
    if let cut = clip["cut"] as? [String: Any], let first = cut["first_samples"] as? Int {
      samples = Array(samples.prefix(first))
    }
    var windows: [JSON] = []
    for case let window as [String: Any] in clip["windows"] as! [Any] {
      let start = window["start"] as! Int
      let count = window["samples"] as! Int
      let slice = Array(samples[start..<min(start + count, samples.count)])
      do {
        let (vector, captures) = try await captured { try await audio.embed(window: slice) }
        windows.append(
          .object([
            ("start", .int(start)),
            ("samples", .int(slice.count)),
            ("captures", .array(captures.map(\.json))),
            ("vector", .vector(vector)),
          ]))
      } catch {
        refuse("audio", "\(id)@\(start)", error)
      }
    }
    audioItems.append(
      .object([
        ("id", .string(id)),
        ("file", .string(clip["file"] as! String)),
        ("file_sha256", .string(fileSha)),
        ("file_sha256_matches_golden", .bool(fileSha == (clip["file_sha256"] as! String))),
        ("samples", .int(samples.count)),
        ("decoder_matches_riff_reader", decoderMatchesRiff.map { .bool($0) } ?? .null),
        ("windows", .array(windows)),
      ]))
  } catch {
    refuse("audio", id, error)
  }
}
print("audio: \(audioItems.count) clips, \(audioItems.reduce(0) { $0 + windowCount($1) }) windows")

func windowCount(_ clip: JSON) -> Int {
  guard case .object(let fields) = clip else { return 0 }
  for (key, value) in fields where key == "windows" {
    if case .array(let windows) = value { return windows.count }
  }
  return 0
}

// MARK: - Vision

/// Pixel `(x, y)`, channel `c`: coremlit's `_corpus.pattern_pixel`.
func patternPixel(_ x: Int, _ y: Int, _ channel: Int) -> UInt8 {
  UInt8((x * 7 + y * 13 + channel * 101 + ((x / 5) ^ (y / 3)) * 29) % 256)
}

func patternBytes(width: Int, height: Int) -> [UInt8] {
  var bytes = [UInt8](repeating: 0, count: width * height * 4)
  for y in 0..<height {
    for x in 0..<width {
      let at = (y * width + x) * 4
      for channel in 0..<3 { bytes[at + channel] = patternPixel(x, y, channel) }
    }
  }
  return bytes
}

func makeImage(_ bytes: [UInt8], width: Int, height: Int, space: CGColorSpace) -> CGImage {
  let provider = CGDataProvider(data: Data(bytes) as CFData)!
  return CGImage(
    width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4, space: space,
    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider, decode: nil,
    shouldInterpolate: false, intent: .defaultIntent)!
}

/// A copy of FluidUse's private `EmbeddingGemma2Vision.rgba` (the redraw it
/// applies to every image), so the probe can measure what it does to a pixel.
func fluidUseRedraw(_ image: CGImage, width: Int, height: Int) -> [UInt8] {
  var pixels = [UInt8](repeating: 0, count: width * height * 4)
  pixels.withUnsafeMutableBytes { buffer in
    let context = CGContext(
      data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4,
      space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)!
    context.interpolationQuality = .high
    context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
  }
  return pixels
}

let sourceSpaces: [(String, CGColorSpace)] = [
  ("sRGB", CGColorSpace(name: CGColorSpace.sRGB)!),
  ("deviceRGB", CGColorSpaceCreateDeviceRGB()),
  ("genericRGBLinear", CGColorSpace(name: CGColorSpace.genericRGBLinear)!),
  ("displayP3", CGColorSpace(name: CGColorSpace.displayP3)!),
  ("linearSRGB", CGColorSpace(name: CGColorSpace.linearSRGB)!),
]

let budgets: [(EmbeddingGemma2Vision.Budget, Int, Int)] = [
  (.fast, 336, 432), (.balanced, 480, 624), (.detailed, 672, 912),
]
var visionItems: [JSON] = []
var redrawReport: [JSON] = []
for (budget, height, width) in budgets {
  let bytes = patternBytes(width: width, height: height)
  // The redraw check: for each source colour space, how many R, G, B values
  // FluidUse's redraw changes at the image's own size.
  var perSpace: [JSON] = []
  var chosen: CGColorSpace?
  for (name, space) in sourceSpaces {
    let redrawn = fluidUseRedraw(
      makeImage(bytes, width: width, height: height, space: space), width: width, height: height)
    var changed = 0
    var worst = 0
    for pixel in 0..<(width * height) {
      for channel in 0..<3 {
        let delta = abs(Int(bytes[pixel * 4 + channel]) - Int(redrawn[pixel * 4 + channel]))
        if delta != 0 { changed += 1 }
        worst = max(worst, delta)
      }
    }
    perSpace.append(
      .object([
        ("source_space", .string(name)), ("values_changed", .int(changed)), ("max_abs_delta", .int(worst)),
      ]))
    if changed == 0 && chosen == nil { chosen = space }
  }
  redrawReport.append(
    .object([("budget", .int(budget.rawValue)), ("height", .int(height)), ("width", .int(width)),
      ("per_source_space", .array(perSpace))]))
  guard let space = chosen else {
    refuse("image", "\(budget.rawValue)", EmbeddingGemma2Error.invalidAsset("no source colour space survives the redraw"))
    continue
  }
  let image = makeImage(bytes, width: width, height: height, space: space)
  do {
    let (vector, captures) = try await captured { try await vision.embed(image: image, budget: budget) }
    visionItems.append(
      .object([
        ("id", .string("pattern-\(budget.rawValue)")),
        ("budget", .int(budget.rawValue)),
        ("height", .int(height)),
        ("width", .int(width)),
        ("captures", .array(captures.map(\.json))),
        ("vector", .vector(vector)),
      ]))
  } catch {
    refuse("image", "\(budget.rawValue)", error)
  }
}
print("image: \(visionItems.count) embedded")

// MARK: - Dump

let counts = Recorder.shared.selectorCounts.sorted { $0.key < $1.key }
let provenance: JSON = .object([
  ("probe", .string("R2")),
  ("fluiduse_revision", .string(fluidUseRevision)),
  ("swift_version", .string(run("swift", ["--version"]))),
  ("sw_vers", .string(run("sw_vers", []))),
  ("cpu", .string(run("sysctl", ["-n", "machdep.cpu.brand_string"]))),
  ("model_directory", .string(modelDirectory.path)),
  (
    "compute_units",
    .object([
      ("text", .string(computeUnitsName(textUnits))),
      ("audio", .string(computeUnitsName(audioUnits))),
      ("vision", .string(computeUnitsName(visionUnits))),
    ])
  ),
  (
    "compute_units_observed",
    .object(
      Recorder.shared.observedComputeUnits.sorted { $0.key < $1.key }.map {
        ($0.key, .string(computeUnitsName(MLComputeUnits(rawValue: $0.value)!)))
      })
  ),
  ("hooked_methods", .array(Hook.installed.map { .string($0) })),
  ("selector_calls", .object(counts.map { ($0.key, .int($0.value)) })),
  ("coregraphics_redraw", .array(redrawReport)),
])

let dump: JSON = .object([
  ("provenance", provenance),
  ("refused", .array(refused)),
  ("text", .array(textItems)),
  ("audio", .array(audioItems)),
  ("image", .array(visionItems)),
])
try dump.render().write(toFile: outputPath, atomically: true, encoding: .utf8)
print("wrote \(outputPath)")
