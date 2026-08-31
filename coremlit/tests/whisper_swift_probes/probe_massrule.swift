// H1 probe: pin the exact numeric semantics of the BNNS f16 pipeline that
// argmax-oss-swift@dcf3a00 LogitsFilter.swift:144-242 runs for the
// timestamp-mass rule:
//   1. BNNS.applyActivation(.logSoftmax) at scalarType Float16 over the full
//      logits vector,
//   2. BNNS.applyReduction(.logSumExp) at Float16 over the timestamp region,
//   3. BNNS.applyReduction(.max) at Float16 over the text region,
//   4. compare the two Float16 scalars with `>`.
// Questions answered:
//   Q1  does the pipeline even run at Float16 on this OS, or error (oracle
//       swallows errors and returns `false`)?
//   Q2  is BNNS's internal arithmetic f32 (round-to-f16 only at outputs) or
//       genuine f16 accumulation? (decisive crafted inputs)
//   Q3  does the logSumExp reduction subtract the max (stable) or not?
//   Q4  which straightforward emulation matches bit-for-bit, at what rate?
//       (random sweeps + near-margin scans)
import Accelerate
import Foundation

typealias F16 = Float16

func hex(_ x: F16) -> String { String(format: "0x%04x", x.bitPattern) }
func show(_ x: F16) -> String { "\(x)(\(hex(x)))" }

// ---------------------------------------------------------------------
// BNNS mirrors (exact API shapes from the oracle)
// ---------------------------------------------------------------------

var bnnsErrorCount = 0

func bnnsLogSoftmax(_ input: [F16]) -> [F16] {
  var inp = input
  let n = inp.count
  var out = [F16](repeating: F16.nan, count: n)
  inp.withUnsafeMutableBufferPointer { ib in
    let inDesc = BNNSNDArrayDescriptor(
      data: UnsafeMutableRawBufferPointer(ib),
      scalarType: F16.self,
      shape: .vector(n, stride: 1)
    )!
    let outDesc = BNNSNDArrayDescriptor.allocateUninitialized(
      scalarType: F16.self,
      shape: .vector(n, stride: 1)
    )
    defer { outDesc.deallocate() }
    do {
      try BNNS.applyActivation(
        activation: BNNS.ActivationFunction.logSoftmax,
        input: inDesc,
        output: outDesc,
        batchSize: 1
      )
      out = outDesc.makeArray(of: F16.self)!
    } catch {
      print("BNNS logSoftmax ERROR (n=\(n)): \(error)")
      bnnsErrorCount += 1
    }
  }
  return out
}

func bnnsReduce(_ fn: BNNS.ReductionFunction, _ input: [F16], label: String) -> F16 {
  var inp = input
  let n = inp.count
  var result = F16.nan
  inp.withUnsafeMutableBufferPointer { ib in
    let inDesc = BNNSNDArrayDescriptor(
      data: UnsafeMutableRawBufferPointer(ib),
      scalarType: F16.self,
      shape: .vector(n, stride: 1)
    )!
    let outDesc = BNNSNDArrayDescriptor.allocateUninitialized(
      scalarType: F16.self,
      shape: .vector(1, stride: 1)
    )
    defer { outDesc.deallocate() }
    do {
      try BNNS.applyReduction(fn, input: inDesc, output: outDesc, weights: nil)
      result = outDesc.makeArray(of: F16.self)![0]
    } catch {
      print("BNNS reduce \(label) ERROR (n=\(n)): \(error)")
      bnnsErrorCount += 1
    }
  }
  return result
}

struct MassResult {
  let ts: F16
  let mx: F16
  var fires: Bool { ts > mx }
}

func bnnsMassRule(_ logits: [F16], timeBegin: Int) -> MassResult {
  let lp = bnnsLogSoftmax(logits)
  let ts = bnnsReduce(.logSumExp, Array(lp[timeBegin...]), label: "logSumExp")
  let mx = bnnsReduce(.max, Array(lp[..<timeBegin]), label: "max")
  return MassResult(ts: ts, mx: mx)
}

// ---------------------------------------------------------------------
// Emulation candidates
// ---------------------------------------------------------------------

// LS-A: f32 sequential; per-element out = f16(v - m - l)   (subtract twice)
func emuLogSoftmaxA(_ input: [F16]) -> [F16] {
  var m = -Float.infinity
  for x in input { m = max(m, Float(x)) }
  var s: Float = 0
  for x in input { s += expf(Float(x) - m) }
  let l = logf(s)
  return input.map { F16(Float($0) - m - l) }
}

// LS-B: f32 sequential; logZ = m + logf(s); out = f16(v - logZ)
func emuLogSoftmaxB(_ input: [F16]) -> [F16] {
  var m = -Float.infinity
  for x in input { m = max(m, Float(x)) }
  var s: Float = 0
  for x in input { s += expf(Float(x) - m) }
  let logZ = m + logf(s)
  return input.map { F16(Float($0) - logZ) }
}

// LS-C: pure f16 sequential (every intermediate rounded to f16)
func emuLogSoftmaxC(_ input: [F16]) -> [F16] {
  var m = -F16.infinity
  for x in input { m = max(m, x) }
  var s: F16 = 0
  for x in input {
    let d = F16(Float(x) - Float(m))  // f16 subtract
    s += F16(expf(Float(d)))          // f16 exp, f16 accumulate
  }
  let l = F16(logf(Float(s)))
  return input.map { x in
    let d = F16(Float(x) - Float(m))
    return F16(Float(d) - Float(l))
  }
}

// LSE-1: f32 sequential, max-subtracted; out = f16(m + logf(s))
func emuLSEStable(_ input: [F16]) -> F16 {
  var m = -Float.infinity
  for x in input { m = max(m, Float(x)) }
  if m == -Float.infinity { return -F16.infinity }
  var s: Float = 0
  for x in input { s += expf(Float(x) - m) }
  return F16(m + logf(s))
}

// LSE-2: f32 sequential, no max subtraction; out = f16(logf(sum exp))
func emuLSENaive(_ input: [F16]) -> F16 {
  var s: Float = 0
  for x in input { s += expf(Float(x)) }
  return F16(logf(s))
}

// LSE-3: pure f16 sequential, max-subtracted
func emuLSEF16(_ input: [F16]) -> F16 {
  var m = -F16.infinity
  for x in input { m = max(m, x) }
  if m == -F16.infinity { return -F16.infinity }
  var s: F16 = 0
  for x in input {
    let d = F16(Float(x) - Float(m))
    s += F16(expf(Float(d)))
  }
  return F16(Float(m) + Float(F16(logf(Float(s)))))
}

func emuMax(_ input: [F16]) -> F16 {
  var m = -F16.infinity
  for x in input { m = max(m, x) }
  return m
}

// Combined emulated mass rules (logSoftmax variant x LSE variant)
func emuMassRule(_ logits: [F16], timeBegin: Int, ls: ([F16]) -> [F16], lse: ([F16]) -> F16)
  -> MassResult
{
  let lp = ls(logits)
  let ts = lse(Array(lp[timeBegin...]))
  let mx = emuMax(Array(lp[..<timeBegin]))
  return MassResult(ts: ts, mx: mx)
}

// ---------------------------------------------------------------------
// Seeded RNG (SplitMix64)
// ---------------------------------------------------------------------

struct SplitMix64 {
  var state: UInt64
  init(seed: UInt64) { state = seed }
  mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }
  mutating func uniform(_ lo: Float, _ hi: Float) -> Float {
    let u = Float(next() >> 40) / Float(1 << 24)
    return lo + (hi - lo) * u
  }
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------

@main
struct Probe {
  static func main() {
    print("=== Q1/Q2 decisive: logSoftmax internal precision ===")
    // v[0]=8, v[1...1100]=-3.
    // f32-internal:   lp[0] = -(logf(1 + 1100*expf(-11))) ~= -0.0182047 -> distinct f16
    // f16-sequential: denominator accumulation loses every exp(-11) -> lp[0] = 0.0 or -0.0
    var d1 = [F16](repeating: -3, count: 1101)
    d1[0] = 8
    let d1Bnns = bnnsLogSoftmax(d1)
    let d1A = emuLogSoftmaxA(d1)
    let d1B = emuLogSoftmaxB(d1)
    let d1C = emuLogSoftmaxC(d1)
    print("lp[0]:  bnns=\(show(d1Bnns[0]))  A=\(show(d1A[0]))  B=\(show(d1B[0]))  C=\(show(d1C[0]))")
    print("lp[1]:  bnns=\(show(d1Bnns[1]))  A=\(show(d1A[1]))  B=\(show(d1B[1]))  C=\(show(d1C[1]))")
    print("lp[1100]: bnns=\(show(d1Bnns[1100]))  A=\(show(d1A[1100]))")

    print("=== Q2 decisive: logSumExp reduction internal precision ===")
    // input [8, -3 x1100]: f32-internal -> f16(8.0182047)=8.015625 ; f16-sequential -> 8.0
    var d2 = [F16](repeating: -3, count: 1101)
    d2[0] = 8
    let d2b = bnnsReduce(.logSumExp, d2, label: "logSumExp")
    print("LSE([8,-3x1100]): bnns=\(show(d2b)) stable_f32=\(show(emuLSEStable(d2))) naive_f32=\(show(emuLSENaive(d2))) f16seq=\(show(emuLSEF16(d2)))")

    print("=== Q3 decisive: max subtraction in logSumExp ===")
    // all -110: stable -> -110 + logf(1101) = -102.996 -> f16 -103.0 ; naive -> logf(0) = -inf
    let d3 = [F16](repeating: -110, count: 1101)
    let d3b = bnnsReduce(.logSumExp, d3, label: "logSumExp")
    print("LSE([-110 x1101]): bnns=\(show(d3b)) stable_f32=\(show(emuLSEStable(d3))) naive_f32=\(show(emuLSENaive(d3)))")

    print("=== -inf handling ===")
    let d4: [F16] = [-F16.infinity, -F16.infinity, 0.5, -F16.infinity, 1.25]
    print("LSE([-inf,-inf,0.5,-inf,1.25]): bnns=\(show(bnnsReduce(.logSumExp, d4, label: "logSumExp"))) stable_f32=\(show(emuLSEStable(d4)))")
    let d5 = [F16](repeating: -F16.infinity, count: 7)
    print("LSE(all -inf x7): bnns=\(show(bnnsReduce(.logSumExp, d5, label: "logSumExp"))) stable_f32=\(show(emuLSEStable(d5)))")
    print("MAX(all -inf x7): bnns=\(show(bnnsReduce(.max, d5, label: "max"))) emu=\(show(emuMax(d5)))")
    var d6 = [F16](repeating: 1.0, count: 64)
    for i in 0..<32 { d6[i] = -F16.infinity }
    let d6b = bnnsLogSoftmax(d6)
    print("logSoftmax with -inf entries: bnns lp[0]=\(show(d6b[0])) lp[63]=\(show(d6b[63])) A lp[0]=\(show(emuLogSoftmaxA(d6)[0])) A lp[63]=\(show(emuLogSoftmaxA(d6)[63]))")

    print("=== reduction size ladder (blocking/order visibility) ===")
    var rng0 = SplitMix64(seed: 7)
    for n in [3, 8, 9, 16, 17, 64, 100, 1501] {
      var v = [F16](repeating: 0, count: n)
      for i in 0..<n { v[i] = F16(rng0.uniform(-12, 0)) }
      let b = bnnsReduce(.logSumExp, v, label: "logSumExp")
      let s = emuLSEStable(v)
      let f = emuLSEF16(v)
      let flag = (b == s) ? "==f32stable" : ((b == f) ? "==f16seq" : "NEITHER")
      print("n=\(n): bnns=\(show(b)) f32stable=\(show(s)) f16seq=\(show(f))  [\(flag)]")
    }

    print("=== SWEEP 1: reduction-only, random logprob vectors ===")
    var rng = SplitMix64(seed: 0xC0FFEE)
    for n in [7, 32, 251, 1501] {
      var matchStable = 0
      var matchNaive = 0
      var matchF16 = 0
      var total = 0
      var firstMismatch = ""
      let iters = 3000
      for it in 0..<iters {
        var v = [F16](repeating: 0, count: n)
        for i in 0..<n { v[i] = F16(rng.uniform(-16, 0)) }
        // sprinkle -inf like a masked region
        if it % 3 == 0 {
          for i in 0..<(n / 4) { v[i] = -F16.infinity }
        }
        let b = bnnsReduce(.logSumExp, v, label: "logSumExp")
        if b == emuLSEStable(v) { matchStable += 1 }
        else if firstMismatch.isEmpty {
          firstMismatch = "it=\(it) bnns=\(show(b)) stable=\(show(emuLSEStable(v)))"
        }
        if b == emuLSENaive(v) { matchNaive += 1 }
        if b == emuLSEF16(v) { matchF16 += 1 }
        total += 1
      }
      print("n=\(n): stable_f32 \(matchStable)/\(total)  naive_f32 \(matchNaive)/\(total)  f16seq \(matchF16)/\(total)  first_stable_mismatch: \(firstMismatch)")
    }

    print("=== SWEEP 2: full mass rule, mid-size (n=4096, tb=3000) ===")
    runMassSweep(n: 4096, timeBegin: 3000, iters: 1500, seedv: 0xBEEF)

    print("=== SWEEP 3: full mass rule, real shape (n=51865, tb=50364) ===")
    runMassSweep(n: 51865, timeBegin: 50364, iters: 300, seedv: 0xFEED)

    print("=== NEAR-MARGIN SCAN: f16-ulp steps of one text logit across the flip ===")
    marginScan()

    print("=== logSoftmax overflow: is it max-subtracted? ===")
    // [88 x4]: expf(88)=1.65e38 finite, sum 6.6e38 OVERFLOWS f32 -> naive lp = -inf;
    // stable lp = -ln(4) = -1.386
    let o1 = [F16](repeating: 88, count: 4)
    print("logSoftmax([88 x4]): bnns lp[0]=\(show(bnnsLogSoftmax(o1)[0])) emuA(stable)=\(show(emuLogSoftmaxA(o1)[0]))")
    let o2 = [F16](repeating: 100, count: 4)  // expf(100)=inf directly
    print("logSoftmax([100 x4]): bnns lp[0]=\(show(bnnsLogSoftmax(o2)[0])) emuA(stable)=\(show(emuLogSoftmaxA(o2)[0]))")

    print("=== .max mixed -inf sanity + NaN semantics ===")
    let mx1: [F16] = [-F16.infinity, -3, -F16.infinity]
    print("MAX([-inf,-3,-inf]): bnns=\(show(bnnsReduce(.max, mx1, label: "max"))) emu=\(show(emuMax(mx1)))")
    let nan1: [F16] = [1.0, F16.nan, 3.0, 2.0]
    print("MAX([1,nan,3,2]): bnns=\(show(bnnsReduce(.max, nan1, label: "max")))")
    print("LSE([1,nan,3,2]): bnns=\(show(bnnsReduce(.logSumExp, nan1, label: "logSumExp")))")
    let nan2: [F16] = [1.0, F16.nan, 3.0, 2.0, -1.0, 0.5, 0.25, -2.0]
    let nanLS = bnnsLogSoftmax(nan2)
    print("logSoftmax([1,nan,3,...]): bnns lp[0]=\(show(nanLS[0])) lp[1]=\(show(nanLS[1])) lp[2]=\(show(nanLS[2]))")

    print("=== DUMP: bit-pinned vectors for Rust libm transfer check ===")
    // V3: 251 f16 values generated purely from bit patterns (no transcendentals),
    // finite-only, magnitudes < 16: pattern = 0x2C00 + (i * 37) % 0x1000, sign from i%3==0
    var v3 = [F16](repeating: 0, count: 251)
    for i in 0..<251 {
      var bits = UInt16(0x2C00 + (i * 37) % 0x1000)
      if i % 3 == 0 { bits |= 0x8000 }
      v3[i] = F16(bitPattern: bits)
    }
    let v3A = emuLogSoftmaxA(v3)
    let v3B = bnnsLogSoftmax(v3)
    print("V3 emuA lp[0]=\(hex(v3A[0])) lp[17]=\(hex(v3A[17])) lp[250]=\(hex(v3A[250]))")
    print("V3 bnns lp[0]=\(hex(v3B[0])) lp[17]=\(hex(v3B[17])) lp[250]=\(hex(v3B[250]))")
    let v3LseA = emuLSENaive(Array(v3A[100...]))
    let v3LseB = bnnsReduce(.logSumExp, Array(v3B[100...]), label: "logSumExp")
    let v3MaxA = emuMax(Array(v3A[..<100]))
    let v3MaxB = bnnsReduce(.max, Array(v3B[..<100]), label: "max")
    print("V3 emuA lse=\(hex(v3LseA)) max=\(hex(v3MaxA)); bnns lse=\(hex(v3LseB)) max=\(hex(v3MaxB))")

    print("=== done; bnnsErrorCount=\(bnnsErrorCount) ===")
  }

  static func buildVector(_ rng: inout SplitMix64, n: Int, timeBegin: Int) -> [F16] {
    var v = [F16](repeating: 0, count: n)
    for i in 0..<n { v[i] = F16(rng.uniform(-22, 4)) }
    // a few text peaks
    for _ in 0..<6 {
      let idx = Int(rng.next() % UInt64(timeBegin))
      v[idx] = F16(rng.uniform(4, 14))
    }
    // one plausible timestamp peak
    let tsIdx = timeBegin + Int(rng.next() % UInt64(n - timeBegin))
    v[tsIdx] = F16(rng.uniform(2, 12))
    // sometimes mask leading timestamps like the timestamp rules do
    if rng.next() % 2 == 0 {
      let maskEnd = timeBegin + Int(rng.next() % UInt64(n - timeBegin))
      for i in timeBegin..<maskEnd where i != tsIdx { v[i] = -F16.infinity }
    }
    return v
  }

  // shift timestamp region so the margin straddles 0, then jitter
  static func tuneMargin(_ v: inout [F16], timeBegin: Int, jitter: Float) {
    var mT = -Float.infinity
    for i in 0..<timeBegin { mT = max(mT, Float(v[i])) }
    // crude f32 estimate of current ts mass (log domain, unnormalized)
    var m = -Float.infinity
    for i in timeBegin..<v.count { m = max(m, Float(v[i])) }
    if m == -Float.infinity { return }
    var s: Float = 0
    for i in timeBegin..<v.count { s += expf(Float(v[i]) - m) }
    let tsLog = m + logf(s)
    let shift = (mT - tsLog) + jitter
    for i in timeBegin..<v.count {
      let x = Float(v[i])
      if x.isFinite { v[i] = F16(x + shift) }
    }
  }

  static func runMassSweep(n: Int, timeBegin: Int, iters: Int, seedv: UInt64) {
    var rng = SplitMix64(seed: seedv)
    var scalarTsA = 0
    var scalarMxA = 0
    var boolAA = 0
    var boolBB = 0
    var boolCC = 0
    var fireCount = 0
    var total = 0
    var mismatches: [String] = []
    for it in 0..<iters {
      var v = buildVector(&rng, n: n, timeBegin: timeBegin)
      tuneMargin(&v, timeBegin: timeBegin, jitter: rng.uniform(-0.06, 0.06))
      let b = bnnsMassRule(v, timeBegin: timeBegin)
      let eA = emuMassRule(v, timeBegin: timeBegin, ls: emuLogSoftmaxA, lse: emuLSEStable)
      let eB = emuMassRule(v, timeBegin: timeBegin, ls: emuLogSoftmaxB, lse: emuLSEStable)
      let eC = emuMassRule(v, timeBegin: timeBegin, ls: emuLogSoftmaxC, lse: emuLSEF16)
      if b.ts == eA.ts { scalarTsA += 1 }
      if b.mx == eA.mx { scalarMxA += 1 }
      if b.fires == eA.fires { boolAA += 1 } else if mismatches.count < 4 {
        mismatches.append(
          "it=\(it) bnns(ts=\(show(b.ts)),mx=\(show(b.mx)),fires=\(b.fires)) emuA(ts=\(show(eA.ts)),mx=\(show(eA.mx)),fires=\(eA.fires))"
        )
      }
      if b.fires == eB.fires { boolBB += 1 }
      if b.fires == eC.fires { boolCC += 1 }
      if b.fires { fireCount += 1 }
      total += 1
    }
    print(
      "n=\(n) tb=\(timeBegin) iters=\(total) fires=\(fireCount): scalar ts==A \(scalarTsA)/\(total), mx==A \(scalarMxA)/\(total); bool A \(boolAA)/\(total), B \(boolBB)/\(total), C(f16) \(boolCC)/\(total)"
    )
    for m in mismatches { print("  MISMATCH \(m)") }
  }

  static func marginScan() {
    // small vector for exhaustive local scan; text region 0..<8, ts 8..<16.
    // ts mass (unnormalized) = -2.25 + ln 8 = -0.1706; fires while the max
    // text logit v[3] < -0.1706, so the flip sits near f16 -0.1706.
    let timeBegin = 8
    var base = [F16](repeating: -6, count: 16)
    for i in timeBegin..<16 { base[i] = -2.25 }
    var flipsB: [UInt16] = []
    var flipsA: [UInt16] = []
    var prevB: Bool? = nil
    var prevA: Bool? = nil
    var x = F16(-0.4)
    for _ in 0..<8000 {
      var v = base
      v[3] = x
      let rb = bnnsMassRule(v, timeBegin: timeBegin).fires
      let ra = emuMassRule(v, timeBegin: timeBegin, ls: emuLogSoftmaxA, lse: emuLSENaive).fires
      if let p = prevB, p != rb { flipsB.append(x.bitPattern) }
      if let p = prevA, p != ra { flipsA.append(x.bitPattern) }
      prevB = rb
      prevA = ra
      x = x.nextUp
      if x > 0.4 { break }
    }
    print("flip points (text-logit bitPattern) bnns=\(flipsB.map { String(format: "0x%04x", $0) })")
    print("flip points (text-logit bitPattern) emuA=\(flipsA.map { String(format: "0x%04x", $0) })")
    // a second, denser configuration: 1501-wide timestamp region like real
    let tb2 = 40
    var base2 = [F16](repeating: -8, count: tb2 + 1501)
    for i in tb2..<base2.count { base2[i] = -9.5 }  // mass ~ -9.5 + ln 1501 = -2.186
    var flipsB2: [UInt16] = []
    var flipsA2: [UInt16] = []
    var pB2: Bool? = nil
    var pA2: Bool? = nil
    var y = F16(-2.4)
    for _ in 0..<1200 {
      var v = base2
      v[7] = y
      let rb = bnnsMassRule(v, timeBegin: tb2).fires
      let ra = emuMassRule(v, timeBegin: tb2, ls: emuLogSoftmaxA, lse: emuLSENaive).fires
      if let p = pB2, p != rb { flipsB2.append(y.bitPattern) }
      if let p = pA2, p != ra { flipsA2.append(y.bitPattern) }
      pB2 = rb
      pA2 = ra
      y = y.nextUp
      if y > -1.9 { break }
    }
    print("scan2 flips bnns=\(flipsB2.map { String(format: "0x%04x", $0) }) emuA=\(flipsA2.map { String(format: "0x%04x", $0) })")
  }
}
