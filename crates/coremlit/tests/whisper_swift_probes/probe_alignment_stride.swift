// H6 probe (coremlit issue #41): the MLMultiArray row pitch that
// `SegmentSeeker.addWordTimestamps`' alignment-gather memcpy ignores.
//
//   swiftc -O -parse-as-library probe_alignment_stride.swift -o probe_alignment_stride
//   ./probe_alignment_stride
//
// WhisperKit allocates every Float16 MLMultiArray through
// `MLMultiArray(shape:dataType:initialValue:)`
// (ArgmaxCore/MLMultiArrayExtensions.swift:11-53), which for `.float16` backs
// the array with an IOSurface CVPixelBuffer (`:121-136`,
// kCVPixelFormatType_OneComponent16Half). CoreVideo pads each row; the plain
// `MLMultiArray(shape:dataType:)` initializer does not. `addWordTimestamps`
// (SegmentSeeker.swift:444-461) binds both stride arrays and then indexes with
// `columnCount` instead, so the gather is only correct while pitch == columns.

import CoreML
import CoreVideo
import Foundation

@main
struct Probe {
  static func pixelBufferBacked(_ rows: Int, _ cols: Int) -> MLMultiArray? {
    var pb: CVPixelBuffer?
    let rc = CVPixelBufferCreate(
      kCFAllocatorDefault, cols, rows, kCVPixelFormatType_OneComponent16Half,
      [kCVPixelBufferIOSurfacePropertiesKey: [:]] as CFDictionary, &pb)
    guard rc == kCVReturnSuccess, let pb else { return nil }
    return MLMultiArray(pixelBuffer: pb, shape: [rows as NSNumber, cols as NSNumber])
  }

  static func main() {
    print("== Float16, pixel-buffer backed (WhisperKit's initialValue: path) ==")
    print("rows cols | shape strides count  storage_rows*pitch")
    for (rows, cols) in [(224, 1500), (120, 1500), (31, 1500), (2, 1500), (1, 1500),
                         (224, 1496), (224, 1504), (224, 100), (224, 8), (224, 9)] {
      guard let a = pixelBufferBacked(rows, cols) else { print("\(rows) \(cols) | FAILED"); continue }
      let st = a.strides.map { $0.intValue }
      print("\(rows) \(cols) | shape=\(a.shape.map { $0.intValue }) strides=\(st) count=\(a.count) storage=\(rows * st[0])")
    }

    print("\n== Float16, plain MLMultiArray(shape:dataType:) ==")
    for (rows, cols) in [(224, 1500), (120, 1500)] {
      guard let a = try? MLMultiArray(shape: [rows as NSNumber, cols as NSNumber], dataType: .float16)
      else { continue }
      print("\(rows) \(cols) | strides=\(a.strides.map { $0.intValue }) count=\(a.count)")
    }

    // The observable consequence: write logical row r via the true stride, read
    // it back the way addWordTimestamps' memcpy does (pitch = columnCount).
    print("\n== gather mismatch, rows=120 cols=1500 ==")
    guard let src = pixelBufferBacked(120, 1500) else { return }
    let pitch = src.strides[0].intValue
    src.withUnsafeMutableBytes { p, _ in
      let base = p.baseAddress!.assumingMemoryBound(to: Float16.self)
      for i in 0..<(120 * pitch) { base[i] = Float16(0) }
      for r in 0..<120 { for c in 0..<1500 { base[r * pitch + c] = Float16(r) } }
    }
    src.withUnsafeBytes { p in
      let base = p.baseAddress!.assumingMemoryBound(to: Float16.self)
      for r in [0, 1, 2, 118, 119] {
        let viaPitch = (0..<1500).map { Float(base[r * pitch + $0]) }
        let viaCols = (0..<1500).map { Float(base[r * 1500 + $0]) }
        let mismatches = zip(viaPitch, viaCols).filter { $0 != $1 }.count
        print("row \(r): true-stride first/last=\(viaPitch[0])/\(viaPitch[1499])  "
              + "columnCount-stride first/last=\(viaCols[0])/\(viaCols[1499])  mismatched=\(mismatches)")
      }
      // ... and the destination overrun: only storage [0, 120*1500) is ever
      // written by the memcpy, so logical row r reads past it by r*pitch+1500-120*1500.
      for r in [117, 118, 119] {
        let overrun = max(0, r * pitch + 1500 - 120 * 1500)
        print("logical row \(r) reads \(overrun) element(s) past the copied prefix "
              + "(kept columns = \(1500 - overrun))")
      }
    }
  }
}
