// H2 probe: pin Swift WhisperKit argmax tie-break semantics on this host.
// Mirrors argmax-oss-swift@dcf3a00 TokenSampler.swift exactly:
//   macOS 15+ path (line 42-83):  MLTensor(MLShapedArray<Float16>(mlArray)).cast(to: Float.self)
//                                 .argmax(alongAxis: -1) -> shapedArray(of: Int32.self)  [toIntArray]
//   legacy path   (line 182-196): BNNS.applyReduction(.argMax, input: f16 descriptor,
//                                 output: Float [1] descriptor)
import CoreML
import Accelerate
import Foundation

typealias F16 = Float16

func makeMLArrayF16(_ values: [F16]) -> MLMultiArray {
  let n = values.count
  let arr = try! MLMultiArray(shape: [1, 1, NSNumber(value: n)], dataType: .float16)
  let ptr = arr.dataPointer.bindMemory(to: F16.self, capacity: n)
  for i in 0..<n { ptr[i] = values[i] }
  return arr
}

@available(macOS 15.0, *)
func mlTensorArgmax(_ values: [F16]) async -> Int {
  let logits = makeMLArrayF16(values)
  // TokenSampler.swift:45
  let logitsTensor = MLTensor(MLShapedArray<F16>(logits)).cast(to: Float.self)
  // TokenSampler.swift:75
  let nextTokenTensor = logitsTensor.argmax(alongAxis: -1)
  // MLTensorExtensions.swift:14-16 (toIntArray)
  return await nextTokenTensor.shapedArray(of: Int32.self).scalars.map { Int($0) }[0]
}

func bnnsArgmax(_ values: [F16]) -> Int {
  var input = values
  let n = input.count
  var result = -999
  input.withUnsafeMutableBufferPointer { buf in
    let logitsDescriptor = BNNSNDArrayDescriptor(
      data: UnsafeMutableRawBufferPointer(buf),
      scalarType: F16.self,
      shape: .vector(n, stride: 1)
    )!
    let argmaxOutput = BNNSNDArrayDescriptor.allocateUninitialized(
      scalarType: Float.self,
      shape: .vector(1, stride: 1)
    )
    defer { argmaxOutput.deallocate() }
    do {
      try BNNS.applyReduction(
        BNNS.ReductionFunction.argMax,
        input: logitsDescriptor,
        output: argmaxOutput,
        weights: nil
      )
      result = Int(argmaxOutput.makeArray(of: Float.self)![0])
    } catch {
      print("BNNS argMax ERROR: \(error)")
      result = -1
    }
  }
  return result
}

struct ProbeCase {
  let name: String
  let n: Int
  let build: (inout [F16]) -> Void
}

@main
struct Probe {
  static func main() async {
    let vocab = 51865
    let cases: [ProbeCase] = [
      ProbeCase(name: "sanity_single_max_at_7", n: 16) { v in v[7] = 5 },
      ProbeCase(name: "tie_2_and_5_small", n: 16) { v in v[2] = 5; v[5] = 5 },
      ProbeCase(name: "tie_0_and_last_small", n: 16) { v in v[0] = 5; v[15] = 5 },
      ProbeCase(name: "all_equal_zero_small", n: 16) { _ in },
      ProbeCase(name: "adjacent_tie_100_101", n: 1024) { v in v[100] = 5; v[101] = 5 },
      ProbeCase(name: "tie_negative_max_3_900", n: 1024) { v in
        for i in 0..<1024 { v[i] = -10 }
        v[3] = -2; v[900] = -2
      },
      ProbeCase(name: "vocab_tie_100_and_30000", n: vocab) { v in v[100] = 5; v[30000] = 5 },
      ProbeCase(name: "vocab_tie_adjacent_50363_50364", n: vocab) { v in v[50363] = 5; v[50364] = 5 },
      ProbeCase(name: "vocab_all_neginf_except_tie_123_50400", n: vocab) { v in
        for i in 0..<v.count { v[i] = -F16.infinity }
        v[123] = 1.5; v[50400] = 1.5
      },
      ProbeCase(name: "vocab_manyway_tie_every_5000", n: vocab) { v in
        var i = 0
        while i < v.count { v[i] = 7; i += 5000 }
      },
      ProbeCase(name: "vocab_all_equal", n: vocab) { v in
        for i in 0..<v.count { v[i] = 1.0 }
      },
      // signed-zero tie: IEEE == treats -0.0 == +0.0; a first-index argmax
      // using IEEE > keeps index 2; total_cmp-based (coremlit today) would
      // pick 5. Which does Swift keep?
      ProbeCase(name: "signed_zero_tie_neg0_at_2_pos0_at_5", n: 16) { v in
        for i in 0..<16 { v[i] = -1 }
        v[2] = F16(bitPattern: 0x8000)  // -0.0
        v[5] = F16(bitPattern: 0x0000)  // +0.0
      },
      ProbeCase(name: "signed_zero_tie_pos0_at_2_neg0_at_5", n: 16) { v in
        for i in 0..<16 { v[i] = -1 }
        v[2] = F16(bitPattern: 0x0000)  // +0.0
        v[5] = F16(bitPattern: 0x8000)  // -0.0
      },
      ProbeCase(name: "nan_at_4_max_at_7", n: 16) { v in
        v[4] = F16.nan
        v[7] = 5
      },
      ProbeCase(name: "nan_at_0_max_at_7", n: 16) { v in
        v[0] = F16.nan
        v[7] = 5
      },
      ProbeCase(name: "all_nan", n: 16) { v in
        for i in 0..<16 { v[i] = F16.nan }
      },
    ]
    for c in cases {
      var v = [F16](repeating: 0, count: c.n)
      c.build(&v)
      var ml: [Int] = []
      var bn: [Int] = []
      for _ in 0..<3 {
        if #available(macOS 15.0, *) {
          ml.append(await mlTensorArgmax(v))
        }
        bn.append(bnnsArgmax(v))
      }
      print("\(c.name): mltensor=\(ml) bnns=\(bn)")
    }
  }
}
