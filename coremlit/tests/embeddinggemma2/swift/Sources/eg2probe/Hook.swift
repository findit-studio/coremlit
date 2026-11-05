import CoreML
import CryptoKit
import Foundation
import ObjectiveC

/// One input tensor FluidUse handed to Core ML.
struct TensorDigest {
  let name: String
  let shape: [Int]
  let dtype: String
  let sha256: String

  var json: JSON {
    .object([
      ("name", .string(name)),
      ("shape", .array(shape.map { .int($0) })),
      ("dtype", .string(dtype)),
      ("sha256", .string(sha256)),
    ])
  }
}

/// One prediction FluidUse made: which function of which loaded model, and the
/// digest of every `MLMultiArray` input.
struct Capture {
  /// `configuration.functionName` of the model that predicted; nil is the
  /// default function (the audio model).
  let functionName: String?
  let inputs: [TensorDigest]

  var json: JSON {
    .object([
      ("function", functionName.map { .string($0) } ?? .null),
      ("inputs", .array(inputs.map { $0.json })),
    ])
  }
}

/// SHA-256 of an array's elements in LOGICAL row-major order, little-endian.
/// The strides are honoured: a padded or non-row-major layout is gathered
/// element by element, so padding bytes never reach the digest.
func digest(of array: MLMultiArray) -> (dtype: String, sha256: String) {
  precondition(1.littleEndian == 1, "the digest assumes a little-endian host")
  let dtype: String
  let width: Int
  switch array.dataType {
  case .float16: (dtype, width) = ("float16", 2)
  case .float32: (dtype, width) = ("float32", 4)
  case .int32: (dtype, width) = ("int32", 4)
  case .double: (dtype, width) = ("float64", 8)
  default: fatalError("unsupported MLMultiArray data type \(array.dataType.rawValue)")
  }
  let shape = array.shape.map(\.intValue)
  let strides = array.strides.map(\.intValue)
  let count = shape.reduce(1, *)
  var rowMajor = [Int](repeating: 1, count: shape.count)
  if shape.count > 1 {
    for axis in stride(from: shape.count - 2, through: 0, by: -1) {
      rowMajor[axis] = rowMajor[axis + 1] * shape[axis + 1]
    }
  }
  let base = UnsafeRawPointer(array.dataPointer)
  var hasher = SHA256()
  // A dimension of extent 1 never contributes its stride.
  let contiguous = zip(zip(shape, strides), rowMajor).allSatisfy { $0.0.0 == 1 || $0.0.1 == $0.1 }
  if contiguous {
    hasher.update(bufferPointer: UnsafeRawBufferPointer(start: base, count: count * width))
  } else {
    var gathered = [UInt8](repeating: 0, count: count * width)
    var index = [Int](repeating: 0, count: shape.count)
    for element in 0..<count {
      var offset = 0
      for axis in 0..<shape.count { offset += index[axis] * strides[axis] }
      gathered.withUnsafeMutableBytes { target in
        target.baseAddress!.advanced(by: element * width)
          .copyMemory(from: base.advanced(by: offset * width), byteCount: width)
      }
      var axis = shape.count - 1
      while axis >= 0 {
        index[axis] += 1
        if index[axis] < shape[axis] { break }
        index[axis] = 0
        axis -= 1
      }
    }
    gathered.withUnsafeBytes { hasher.update(bufferPointer: $0) }
  }
  return (dtype, hasher.finalize().map { String(format: "%02x", $0) }.joined())
}

/// Collects the predictions made between `reset()` calls.
final class Recorder: @unchecked Sendable {
  static let shared = Recorder()

  private let lock = NSLock()
  private var captures: [Capture] = []
  /// Providers already recorded in this item (kept alive so an address is
  /// never reused): a prediction reaches several selectors on some paths, and
  /// must be recorded once.
  private var seen: [ObjectIdentifier: AnyObject] = [:]
  private(set) var selectorCounts: [String: Int] = [:]
  private(set) var observedComputeUnits: [String: Int] = [:]

  func reset() {
    lock.lock()
    defer { lock.unlock() }
    captures = []
    seen = [:]
  }

  func take() -> [Capture] {
    lock.lock()
    defer { lock.unlock() }
    return captures
  }

  func record(model: AnyObject, provider: AnyObject, selector: String) {
    lock.lock()
    defer { lock.unlock() }
    selectorCounts[selector, default: 0] += 1
    if seen[ObjectIdentifier(provider)] != nil { return }
    seen[ObjectIdentifier(provider)] = provider
    guard let features = provider as? MLFeatureProvider else { return }
    let functionName = (model as? MLModel)?.configuration.functionName
    if let units = (model as? MLModel)?.configuration.computeUnits {
      observedComputeUnits[functionName ?? "<default>"] = units.rawValue
    }
    var inputs: [TensorDigest] = []
    for name in features.featureNames.sorted() {
      guard let array = features.featureValue(for: name)?.multiArrayValue else { continue }
      let (dtype, sha) = digest(of: array)
      inputs.append(
        TensorDigest(name: name, shape: array.shape.map(\.intValue), dtype: dtype, sha256: sha))
    }
    captures.append(Capture(functionName: functionName, inputs: inputs))
  }
}

@_silgen_name("objc_getClassList")
private func rawGetClassList(_ buffer: UnsafeMutablePointer<UnsafeRawPointer?>?, _ count: Int32) -> Int32

/// `class_getName` through `dlsym`: `@_silgen_name` reserves that symbol.
private let rawGetClassName: @convention(c) (UnsafeRawPointer) -> UnsafePointer<CChar> = unsafeBitCast(
  dlsym(UnsafeMutableRawPointer(bitPattern: -2), "class_getName"),
  to: (@convention(c) (UnsafeRawPointer) -> UnsafePointer<CChar>).self)

@_silgen_name("class_getSuperclass")
private func rawGetSuperclass(_ cls: UnsafeRawPointer) -> UnsafeRawPointer?

/// Swizzles the four public `MLModel` predict selectors, and the request-based
/// one `try await model.prediction(from:)` actually reaches, so every
/// prediction passes through `Recorder` first, then runs the original
/// implementation.
///
/// That request-based selector is `submitPredictionRequest:completionHandler:`:
/// CoreML's `prediction(from:options:) async` wraps the input in an
/// `MLGenericPredictionRequest` (which carries `inputFeatures`) and submits it,
/// and none of the four public selectors runs on that path.
enum Hook {
  private static var swizzled = Set<Method>()
  /// `Class.selector` of every method hooked, for the dump's provenance.
  private(set) static var installed: [String] = []

  private static let selectors = [
    "predictionFromFeatures:completionHandler:",
    "predictionFromFeatures:options:completionHandler:",
    "predictionFromFeatures:error:",
    "predictionFromFeatures:options:error:",
    "submitPredictionRequest:completionHandler:",
  ]

  /// Hooks `MLModel` and every subclass that overrides a predict selector.
  /// Safe to call again after models load: a concrete subclass (Core ML loads
  /// models as `MLDelegateModel`) may only exist once a model has been
  /// instantiated.
  static func install() {
    // The class list holds classes that crash when ARC retains them (for
    // example `__NSGenericDeallocHandler`), so it is walked as raw pointers
    // and only the classes that matter are turned into `AnyClass`. `MLModel`
    // is found by name in that list: Swift's `MLModel.self` and
    // `NSClassFromString("MLModel")` are a different object from the class
    // every Core ML subclass inherits from.
    let total = Int(rawGetClassList(nil, 0))
    var classes = [UnsafeRawPointer?](repeating: nil, count: total)
    let filled = Int(rawGetClassList(&classes, Int32(total)))
    let roots = Set(
      classes.prefix(filled).compactMap { $0 }.filter { String(cString: rawGetClassName($0)) == "MLModel" })
    var targets: [UnsafeRawPointer] = []
    for case let candidate? in classes.prefix(filled) {
      var chain: UnsafeRawPointer? = candidate
      while let current = chain {
        if roots.contains(current) {
          targets.append(candidate)
          break
        }
        chain = rawGetSuperclass(current)
      }
    }
    for raw in targets {
      let target = unsafeBitCast(raw, to: AnyClass.self)
      for name in selectors {
        let selector = NSSelectorFromString(name)
        guard let method = ownMethod(of: target, selector: selector) else { continue }
        if swizzled.contains(method) { continue }
        swizzled.insert(method)
        wrap(method, selector: selector, name: name)
        installed.append("\(String(cString: rawGetClassName(raw))).\(name)")
      }
    }
  }

  /// The method `target` itself defines, not an inherited one.
  private static func ownMethod(of target: AnyClass, selector: Selector) -> Method? {
    var count: UInt32 = 0
    guard let methods = class_copyMethodList(target, &count) else { return nil }
    defer { free(methods) }
    for index in 0..<Int(count) where method_getName(methods[index]) == selector {
      return methods[index]
    }
    return nil
  }

  private static func wrap(_ method: Method, selector: Selector, name: String) {
    let original = method_getImplementation(method)
    let block: AnyObject
    switch name {
    case "predictionFromFeatures:completionHandler:":
      typealias Call = @convention(c) (AnyObject, Selector, AnyObject, AnyObject) -> Void
      let call = unsafeBitCast(original, to: Call.self)
      let hooked: @convention(block) (AnyObject, AnyObject, AnyObject) -> Void = { model, input, handler in
        Recorder.shared.record(model: model, provider: input, selector: name)
        call(model, selector, input, handler)
      }
      block = unsafeBitCast(hooked, to: AnyObject.self)
    case "submitPredictionRequest:completionHandler:":
      typealias Call = @convention(c) (AnyObject, Selector, AnyObject, AnyObject) -> Void
      let call = unsafeBitCast(original, to: Call.self)
      let hooked: @convention(block) (AnyObject, AnyObject, AnyObject) -> Void = { model, request, handler in
        // The request holds the features; its `inputFeatures` getter is the
        // only way to reach them.
        if let input = request.perform(NSSelectorFromString("inputFeatures"))?.takeUnretainedValue() {
          Recorder.shared.record(model: model, provider: input, selector: name)
        }
        call(model, selector, request, handler)
      }
      block = unsafeBitCast(hooked, to: AnyObject.self)
    case "predictionFromFeatures:options:completionHandler:":
      typealias Call = @convention(c) (AnyObject, Selector, AnyObject, AnyObject, AnyObject) -> Void
      let call = unsafeBitCast(original, to: Call.self)
      let hooked: @convention(block) (AnyObject, AnyObject, AnyObject, AnyObject) -> Void = {
        model, input, options, handler in
        Recorder.shared.record(model: model, provider: input, selector: name)
        call(model, selector, input, options, handler)
      }
      block = unsafeBitCast(hooked, to: AnyObject.self)
    case "predictionFromFeatures:error:":
      typealias Call =
        @convention(c) (AnyObject, Selector, AnyObject, UnsafeMutableRawPointer?) -> Unmanaged<AnyObject>?
      let call = unsafeBitCast(original, to: Call.self)
      let hooked: @convention(block) (AnyObject, AnyObject, UnsafeMutableRawPointer?) -> Unmanaged<AnyObject>? = {
        model, input, error in
        Recorder.shared.record(model: model, provider: input, selector: name)
        return call(model, selector, input, error)
      }
      block = unsafeBitCast(hooked, to: AnyObject.self)
    default:
      typealias Call =
        @convention(c) (AnyObject, Selector, AnyObject, AnyObject, UnsafeMutableRawPointer?) -> Unmanaged<AnyObject>?
      let call = unsafeBitCast(original, to: Call.self)
      let hooked: @convention(block) (AnyObject, AnyObject, AnyObject, UnsafeMutableRawPointer?) -> Unmanaged<AnyObject>? = {
        model, input, options, error in
        Recorder.shared.record(model: model, provider: input, selector: name)
        return call(model, selector, input, options, error)
      }
      block = unsafeBitCast(hooked, to: AnyObject.self)
    }
    method_setImplementation(method, imp_implementationWithBlock(block))
  }
}
