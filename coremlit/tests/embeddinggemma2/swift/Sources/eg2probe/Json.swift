import Foundation

/// A minimal JSON value with deterministic output. `JSONSerialization` would
/// print floats through `NSNumber` (double precision, not the shortest f32
/// round-trip), so the dump is written by hand.
enum JSON {
  case null
  case bool(Bool)
  case int(Int)
  case float(Float)
  case string(String)
  case array([JSON])
  case object([(String, JSON)])

  /// An f32 vector as JSON numbers: Swift's `Float.description` is the
  /// shortest decimal that round-trips to the same f32.
  static func vector(_ values: [Float]) -> JSON { .array(values.map { .float($0) }) }

  func write(into out: inout String, indent: Int = 0) {
    switch self {
    case .null: out += "null"
    case .bool(let value): out += value ? "true" : "false"
    case .int(let value): out += String(value)
    case .float(let value):
      precondition(value.isFinite, "non-finite value in the dump")
      out += value.description
    case .string(let value): JSON.escape(value, into: &out)
    case .array(let items):
      if items.isEmpty {
        out += "[]"
        return
      }
      // Arrays of scalars stay on one line (vectors); arrays of containers
      // get one element per line.
      let flat = items.allSatisfy {
        switch $0 {
        case .array, .object: return false
        default: return true
        }
      }
      if flat {
        out += "["
        for (index, item) in items.enumerated() {
          if index > 0 { out += ", " }
          item.write(into: &out)
        }
        out += "]"
      } else {
        out += "[\n"
        for (index, item) in items.enumerated() {
          out += String(repeating: "  ", count: indent + 1)
          item.write(into: &out, indent: indent + 1)
          out += index + 1 < items.count ? ",\n" : "\n"
        }
        out += String(repeating: "  ", count: indent) + "]"
      }
    case .object(let fields):
      if fields.isEmpty {
        out += "{}"
        return
      }
      out += "{\n"
      for (index, field) in fields.enumerated() {
        out += String(repeating: "  ", count: indent + 1)
        JSON.escape(field.0, into: &out)
        out += ": "
        field.1.write(into: &out, indent: indent + 1)
        out += index + 1 < fields.count ? ",\n" : "\n"
      }
      out += String(repeating: "  ", count: indent) + "}"
    }
  }

  static func escape(_ text: String, into out: inout String) {
    out += "\""
    for scalar in text.unicodeScalars {
      switch scalar {
      case "\"": out += "\\\""
      case "\\": out += "\\\\"
      case "\n": out += "\\n"
      case "\r": out += "\\r"
      case "\t": out += "\\t"
      default:
        if scalar.value < 0x20 {
          out += String(format: "\\u%04x", scalar.value)
        } else {
          out.unicodeScalars.append(scalar)
        }
      }
    }
    out += "\""
  }

  func render() -> String {
    var out = ""
    write(into: &out)
    out += "\n"
    return out
  }
}
