// The online-clusterer Swift ORACLE for
// `crates/coremlit/tests/speaker/parity_online_swift.rs`.
//
// It `import`s the LOCAL FluidAudio checkout and drives its
// `SpeakerManager.assignSpeaker` DIRECTLY on a deterministic synthetic
// embedding sequence (a documented 64-bit LCG — NOT model output), then dumps
// the per-step decision trace as JSON on stdout. The committed
// `../../fixtures/golden_online_swift/trace.json` is exactly this program's
// stdout, so the Rust gate replays it with no Swift toolchain and no models.
//
// The single source of truth for the sequence + hashing is the Rust harness
// (`tests/speaker/parity_online_swift.rs`): the `Lcg`, `syntheticSequence`, and
// `fnv1aF32` below MIRROR it byte-for-byte. Proof of faithfulness is that the
// FNV-1a-64 hash of the 48×256 raw embeddings reproduces the committed
// `inputFnv1a`; only then is the SECOND, independently Swift-computed hash —
// `durationsFnv1a` over the `[Float]` duration sequence — trustworthy. That
// duration hash is this oracle's whole point: it CROSS-LANGUAGE-attests the
// durations fed to `assignSpeaker`, closing the gap where a Rust self-hash
// could not observe a Swift-side duration divergence (finding M3).
//
// stdout carries ONLY the JSON (so `swift run online_oracle > trace.json`
// works); all diagnostics go to stderr.

import FluidAudio
import Foundation

// ── Generator constants (mirror tests/speaker/parity_online_swift.rs) ──
let SEED: UInt64 = 0xDEAD_BEEF_CAFE_F00D
let STEPS = 48
let PROTOTYPES = 8
let BLOCK = 32
let DIM = 256  // == diaric::embed::EMBEDDING_DIM
let NOISE_SCALE: Float = 0.02
let DUR_BASE: Float = 0.3
let DUR_SPAN: Float = 2.0

// SpeakerManager options — the exact trace thresholds.
let SPEAKER_THRESHOLD: Float = 0.65
let EMBEDDING_THRESHOLD: Float = 0.45
let MIN_SPEECH_DURATION: Float = 1.0

/// The 64-bit LCG (Knuth MMIX constants). Byte-identical to the Rust `Lcg`:
/// `next_u64` advances the state and returns the POST-update value; `next_unit`
/// takes the top 24 bits scaled by an exact `2^-24`; `next_index` truncates
/// `next_unit()*count` toward zero and clamps to `count-1`.
struct Lcg {
  var state: UInt64
  init(seed: UInt64) { state = seed }

  mutating func nextU64() -> UInt64 {
    state = state &* 6364136223846793005 &+ 1442695040888963407
    return state
  }

  /// f32 in `[0, 1)` from the top 24 bits, scaled by an exact `2^-24`
  /// (`1 / 16777216`) — matches Rust's `bits as f32 * (1.0 / 16777216.0)`.
  mutating func nextUnit() -> Float {
    let bits = UInt32(truncatingIfNeeded: nextU64() >> 40)
    return Float(bits) * (1.0 / 16777216.0)
  }

  /// A prototype index in `0..<count` from the high bits — matches Rust's
  /// `((next_unit() * count as f32) as usize).min(count - 1)`.
  mutating func nextIndex(_ count: Int) -> Int {
    return min(Int(nextUnit() * Float(count)), count - 1)
  }
}

/// FNV-1a-64 over the little-endian bytes of an `f32` buffer. Byte-identical to
/// the Rust `common::fnv1a_f32`: offset basis `0xcbf29ce484222325`, prime
/// `0x100000001b3`, each `f32` contributing its 4 little-endian IEEE-754 bytes
/// (least-significant first, matching `f32::to_le_bytes` on little-endian).
func fnv1aF32(_ samples: [Float]) -> UInt64 {
  var h: UInt64 = 0xcbf2_9ce4_8422_2325
  for s in samples {
    let bits = s.bitPattern
    for i in 0..<4 {
      let byte = UInt8(truncatingIfNeeded: bits >> (8 * i))
      h ^= UInt64(byte)
      h = h &* 0x0000_0100_0000_01b3
    }
  }
  return h
}

/// Lowercase, zero-padded 16-hex-digit rendering of a hash (Rust `fnv_hex`).
func fnvHex(_ h: UInt64) -> String {
  let s = String(h, radix: 16)
  return String(repeating: "0", count: max(0, 16 - s.count)) + s
}

/// Regenerate the `(raw embedding, speech duration)` sequence. Byte-identical
/// to the Rust `synthetic_sequence`: per step, 1 index draw, 1 duration unit,
/// then `DIM` noise units, applied over the one-hot prototype block.
func syntheticSequence() -> [(embedding: [Float], duration: Float)] {
  var lcg = Lcg(seed: SEED)
  var seq: [(embedding: [Float], duration: Float)] = []
  seq.reserveCapacity(STEPS)
  for _ in 0..<STEPS {
    let j = lcg.nextIndex(PROTOTYPES)
    let duration = DUR_BASE + lcg.nextUnit() * DUR_SPAN
    var raw = [Float](repeating: 0, count: DIM)
    for d in 0..<DIM {
      let base: Float = (d >= j * BLOCK && d < (j + 1) * BLOCK) ? 1.0 : 0.0
      let noise = (lcg.nextUnit() * 2.0 - 1.0) * NOISE_SCALE
      raw[d] = base + noise
    }
    seq.append((raw, duration))
  }
  return seq
}

// ── The trace JSON schema (matches the committed trace exactly under
//    JSONEncoder `[.prettyPrinted, .sortedKeys]`). ──

/// One recorded decision. `centroid`/`id` are nil (and thus OMITTED by the
/// encoder) for a DROPPED step, giving `{ "kind" : "dropped" }`.
struct Step: Encodable {
  let centroid: [Float]?
  let id: Int?
  let kind: String
}

/// The Swift-attested generator provenance block.
struct Generator: Encodable {
  let blockSize = 32
  let durationBase: Float = 0.3
  let durationSpan: Float = 2.0
  let embeddingDim = 256
  let noiseScale: Float = 0.02
  let prototypes = 8
  let seed = "0xdeadbeefcafef00d"
  let steps = 48
}

/// The SpeakerManager options the trace was generated at.
struct Options: Encodable {
  let embeddingThreshold: Float = 0.45
  let minSpeechDuration: Float = 1.0
  let speakerThreshold: Float = 0.65
}

/// The full committed-trace object.
struct Trace: Encodable {
  let droppedCount: Int
  let durationsFnv1a: String
  let existingCount: Int
  let generator: Generator
  let inputFnv1a: String
  let newCount: Int
  let options: Options
  let steps: [Step]
}

func eprint(_ s: String) {
  FileHandle.standardError.write((s + "\n").data(using: .utf8)!)
}

// ── Drive the oracle. ──
let seq = syntheticSequence()

// Input-match hashes (Step 1 proof). The embedding hash is the faithfulness
// gate: it MUST reproduce the committed inputFnv1a before the duration hash is
// trustworthy.
var flat: [Float] = []
flat.reserveCapacity(STEPS * DIM)
for (embedding, _) in seq { flat.append(contentsOf: embedding) }
let inputFnv1a = fnvHex(fnv1aF32(flat))

let durations = seq.map { $0.duration }
let durationsFnv1a = fnvHex(fnv1aF32(durations))

eprint("[online-oracle] inputFnv1a     = \(inputFnv1a)")
eprint("[online-oracle] durationsFnv1a = \(durationsFnv1a)")

// Drive FluidAudio's SpeakerManager on the RAW embeddings (it L2-normalizes
// internally — matching the Rust side, which hashes+feeds the same raw vectors).
var manager = SpeakerManager(
  speakerThreshold: SPEAKER_THRESHOLD,
  embeddingThreshold: EMBEDDING_THRESHOLD,
  minSpeechDuration: MIN_SPEECH_DURATION
)

var steps: [Step] = []
steps.reserveCapacity(STEPS)
var newCount = 0
var existingCount = 0
var droppedCount = 0

for (embedding, duration) in seq {
  // Snapshot the id set BEFORE the call to classify new-vs-existing.
  let idsBefore = Set(manager.speakerIds)
  let assigned = manager.assignSpeaker(embedding, speechDuration: duration)

  if let speaker = assigned {
    let kind = idsBefore.contains(speaker.id) ? "existing" : "new"
    if kind == "new" { newCount += 1 } else { existingCount += 1 }
    // The trace encodes ids as dense integers from 1 (SpeakerManager assigns
    // `String(nextSpeakerId)` starting at 1); recover the integer form.
    guard let idInt = Int(speaker.id) else {
      eprint("[online-oracle] FATAL: non-integer Speaker.id \"\(speaker.id)\"")
      exit(1)
    }
    steps.append(Step(centroid: speaker.currentEmbedding, id: idInt, kind: kind))
  } else {
    droppedCount += 1
    steps.append(Step(centroid: nil, id: nil, kind: "dropped"))
  }
}

eprint("[online-oracle] new=\(newCount) existing=\(existingCount) dropped=\(droppedCount)")

let trace = Trace(
  droppedCount: droppedCount,
  durationsFnv1a: durationsFnv1a,
  existingCount: existingCount,
  generator: Generator(),
  inputFnv1a: inputFnv1a,
  newCount: newCount,
  options: Options(),
  steps: steps
)

let encoder = JSONEncoder()
encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
let data = try encoder.encode(trace)
FileHandle.standardOutput.write(data)
