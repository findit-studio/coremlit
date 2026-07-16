//! Numerical-guard gate over every shipped CoreML graph.
//!
//! # The defect class
//!
//! A model graph contains a numerically-guarded op — `log`, `sqrt`,
//! `rsqrt`, a normalization, or a pooling divide — whose guard epsilon is
//! **smaller than fp16's smallest subnormal, `2^-24` ≈ 5.96e-8**. In an
//! fp32 conversion the guard survives. Executed in fp16 it **rounds to
//! zero**, the guard goes inert, and the op saturates or divides by zero.
//!
//! The failures are silent, systematic (bit-identical run to run), and
//! surface only on the ANE/GPU paths that actually compute in fp16 —
//! which is every path by default, because
//! [`ComputeUnits::default()`][coremlit::ComputeUnits] is
//! `All`. A model's *declared* MIL dtype is therefore **not** protection:
//! `speakerkit/wespeaker.mlmodelc` is fp32 end-to-end in its MIL and still
//! collapses to a cosine of 0.035 when the same fp32 artifact is loaded
//! `CpuOnly → All`. This gate consequently holds **every** graph to the
//! fp16 floor, whatever dtype it declares.
//!
//! # Why a graph gate and not an output check
//!
//! The graph is the only place the defect is *legible*. Downstream it
//! looks like a slightly-worse DER or a word timing that drifted — a
//! quality regression, not a bug. Three separate crates shipped this and
//! none of their output-level tests caught it. `model.mil` is plain text
//! and states the epsilon literally; this gate reads it and does arithmetic
//! on it. No inference, no models needed to test the checker itself.
//!
//! # What this asserts
//!
//! - Every `.mlmodelc` discovered under `Models/` is parsed — a **walk**,
//!   never a hardcoded list, so a newly-converted or newly-added model is
//!   covered the moment it lands.
//! - Every guard site's *effective* floor (the op's own `epsilon =` plus
//!   any provable lower bound on its input, from an `add`/`clip`/`maximum`
//!   guard) is compared against `2^-24`.
//! - Findings must match [`KNOWN_DEFECTS`] **exactly**. An unpinned model
//!   that grows a vanishing guard fails; a pinned one that is quietly
//!   *repaired* also fails, so a fix cannot land un-noticed either.
//! - A `.mlmodelc` with no readable `model.mil` is a hard failure, and a
//!   pinned defect that has disappeared from an otherwise-present vendor
//!   tree is a hard failure. Nothing silently skips.
//!
//! When `Models/` is absent the sweep is `ignored`, never a green `ok`
//! over zero models (see `build.rs`). The parser tests below are hermetic
//! and always run: they pin the checker against verbatim excerpts of the
//! real known-bad and known-good graphs, so the gate itself cannot rot
//! into something that always passes.

use std::{
  collections::BTreeMap,
  fs, io,
  path::{Path, PathBuf},
};

/// fp16's smallest subnormal, `2^-24`. An epsilon below this is not
/// representable in fp16 and rounds to zero — the guard becomes inert.
const FP16_MIN_SUBNORMAL: f64 = 5.960_464_477_539_063e-8;

/// fp16's smallest *normal*, `2^-14`. Not the gate's threshold; recorded
/// because guards between it and [`FP16_MIN_SUBNORMAL`] survive only as
/// subnormals, which some kernels flush to zero.
#[allow(dead_code)]
const FP16_MIN_NORMAL: f64 = 6.103_515_625e-5;

// ---------------------------------------------------------------------------
// MIL parsing
//
// Hand-rolled over MIL's fixed one-statement-per-line shape, mirroring
// `whisperkit/tests/models_lock.rs`'s hand-rolled lock reader: no parser
// dependency for a grammar this small and this fixed.
//
//   tensor<fp16, [1, 2999, 29]> var_849_cast_fp16 =
//       log(epsilon = var_849_epsilon_0, x = var_849_softmax_cast_fp16)
//       [name = tensor<string, []>("op_849_cast_fp16")];
// ---------------------------------------------------------------------------

/// One parsed MIL statement: `tensor<DTYPE, [..]> VAR = OP(ARGS)[ATTRS];`
struct Stmt {
  dtype: String,
  op: String,
  args: String,
}

/// A parsed graph: scalar constants resolved to values, plus every
/// variable's producing statement.
struct Graph {
  consts: BTreeMap<String, f64>,
  producers: BTreeMap<String, Stmt>,
  /// Statement lines that NAME a guarded op (see [`GUARD_LOOKING_OPS`]) but
  /// did not parse into a resolvable [`Stmt`]. Completeness accounting: a
  /// guard the reader cannot read is a hole, not a pass. [`Graph::audit`]
  /// carries these forward and the sweep fails with the line quoted, so a
  /// re-conversion that emits a guard in syntax this hand-rolled reader does
  /// not yet handle can never masquerade as a clean sweep — the exact way a
  /// partial parse used to stay GREEN with one recognized guard beside a new
  /// vanishing one.
  unresolved: Vec<String>,
}

/// Parses a hex float literal (`0x1p-149`, `0x1.5798eep-27`, `0x0p+0`).
///
/// `f64::from_str` does not accept hex floats, and every epsilon that
/// matters in these graphs is written in exactly that form.
fn parse_hex_float(s: &str) -> Option<f64> {
  let s = s.trim();
  let (neg, s) = match s.strip_prefix('-') {
    Some(rest) => (true, rest),
    None => (false, s.strip_prefix('+').unwrap_or(s)),
  };
  let s = s.strip_prefix("0x").or_else(|| s.strip_prefix("0X"))?;
  let (mantissa, exponent) = s.split_once(['p', 'P'])?;
  let (int_part, frac_part) = mantissa.split_once('.').unwrap_or((mantissa, ""));
  if int_part.is_empty() && frac_part.is_empty() {
    return None;
  }

  let mut value = 0.0_f64;
  for c in int_part.chars() {
    value = value * 16.0 + f64::from(c.to_digit(16)?);
  }
  let mut scale = 1.0 / 16.0;
  for c in frac_part.chars() {
    value += f64::from(c.to_digit(16)?) * scale;
    scale /= 16.0;
  }

  let exp: i32 = exponent.parse().ok()?;
  value *= 2.0_f64.powi(exp);
  Some(if neg { -value } else { value })
}

/// Parses a MIL scalar literal — hex float first, then decimal.
fn parse_scalar(tok: &str) -> Option<f64> {
  let tok = tok.trim();
  if tok.contains("0x") || tok.contains("0X") {
    return parse_hex_float(tok);
  }
  tok.parse::<f64>().ok()
}

/// Splits `a = 1, b = tensor<int32, [1]>([2])` on depth-0 commas.
fn split_args(args: &str) -> Vec<&str> {
  let (mut out, mut depth, mut start) = (Vec::new(), 0_i32, 0_usize);
  for (i, c) in args.char_indices() {
    match c {
      '(' | '[' | '<' => depth += 1,
      ')' | ']' | '>' => depth -= 1,
      ',' if depth == 0 => {
        out.push(args[start..i].trim());
        start = i + 1;
      }
      _ => {}
    }
  }
  let tail = args[start..].trim();
  if !tail.is_empty() {
    out.push(tail);
  }
  out
}

/// Value of the `key = value` argument named `key`, if present.
fn arg<'a>(args: &'a str, key: &str) -> Option<&'a str> {
  split_args(args).into_iter().find_map(|pair| {
    let (k, v) = pair.split_once('=')?;
    (k.trim() == key).then(|| v.trim())
  })
}

/// The op names that make a statement *guard-looking*. If the reader cannot
/// fully parse a statement whose op is one of these, the sweep fails rather
/// than dropping it silently: it may be a numerically-guarded op emitted in
/// syntax this hand-rolled reader does not yet handle — exactly what a new
/// coremltools re-conversion can produce. Covers the guard SITES (`log`,
/// `rsqrt`, `sqrt`, `real_div`, the norms) AND the floor-contributing ops a
/// site's guard is resolved through (`add`, `clip`, `maximum`, `softmax`),
/// because an unreadable `clip` can make a `sqrt` guard vanish just as
/// silently as an unreadable `sqrt`. `exp` is included as the containment op
/// the folded-log audit reasons about. The check is dormant on today's tree
/// (every guard statement parses) and arms only when a re-conversion changes
/// the shape of one — the bias is deliberately toward a loud review over a
/// silent drop.
const GUARD_LOOKING_OPS: &[&str] = &[
  "log",
  "rsqrt",
  "sqrt",
  "real_div",
  "instance_norm",
  "layer_norm",
  "batch_norm",
  "add",
  "clip",
  "maximum",
  "softmax",
  "exp",
];

/// True when `b` can appear inside a MIL identifier.
fn is_ident_byte(b: u8) -> bool {
  b.is_ascii_alphanumeric() || b == b'_'
}

/// The guard op *called* in `line`, if any: a `NAME(` where NAME is in
/// [`GUARD_LOOKING_OPS`] and stands as a whole token — not the tail of a
/// longer identifier such as `catalog(` or `log_softmax(`. Used only on
/// statement lines that FAILED to parse, to tell a completeness hole (a
/// guard we must not lose) from a benign non-guard op we never audited.
fn guard_op_in(line: &str) -> Option<&'static str> {
  let bytes = line.as_bytes();
  GUARD_LOOKING_OPS.iter().copied().find(|&op| {
    let mut from = 0;
    while let Some(rel) = line[from..].find(op) {
      let i = from + rel;
      let after = i + op.len();
      let before_ok = i == 0 || !is_ident_byte(bytes[i - 1]);
      if before_ok && bytes.get(after) == Some(&b'(') {
        return true;
      }
      from = i + 1;
    }
    false
  })
}

/// The outcome of reading one physical line as a MIL statement.
enum ParseOutcome {
  /// Not a `tensor<...>` statement line at all — skipped, as before.
  NotStatement,
  /// A fully-parsed statement: its variable, producing op, and (when the op
  /// is a scalar `const`) the resolved value.
  Parsed {
    var: String,
    stmt: Stmt,
    const_val: Option<f64>,
  },
  /// A `tensor<...>` statement line that did not parse. `guard` names the
  /// guard op it appears to call, if any — `Some` is a completeness hole.
  Unparsed { guard: Option<&'static str> },
}

/// Reads one trimmed line. Any `tensor<...>`-shaped statement that does not
/// parse is reported as [`ParseOutcome::Unparsed`] — never silently skipped —
/// so a guard emitted in unhandled syntax is surfaced, not lost.
fn parse_stmt_line(line: &str) -> ParseOutcome {
  let Some(rest) = line.strip_prefix("tensor<") else {
    return ParseOutcome::NotStatement;
  };
  // From here the line IS a statement; any failure to parse is Unparsed, and
  // a completeness hole iff the raw line names a guard op.
  let unparsed = || ParseOutcome::Unparsed {
    guard: guard_op_in(line),
  };

  // `fp16, [1, 2999, 29]> var = op(args)[attrs];` — shapes never nest angle
  // brackets, so the first `>` closes the tensor type.
  let Some((ty, rest)) = rest.split_once('>') else {
    return unparsed();
  };
  let dtype = ty.split(',').next().unwrap_or("").trim().to_string();

  let Some((var, rest)) = rest.split_once('=') else {
    return unparsed();
  };
  let var = var.trim();
  if var.is_empty() || !var.chars().all(|c| c.is_alphanumeric() || c == '_') {
    return unparsed();
  }

  let rest = rest.trim();
  let Some(open) = rest.find('(') else {
    return unparsed();
  };
  let op = rest[..open].trim().to_string();

  // Balanced scan for the op's argument list.
  let mut depth = 0_i32;
  let mut close = None;
  for (i, c) in rest[open..].char_indices() {
    match c {
      '(' => depth += 1,
      ')' => {
        depth -= 1;
        if depth == 0 {
          close = Some(open + i);
          break;
        }
      }
      _ => {}
    }
  }
  let Some(close) = close else {
    // The op name is already in hand — classify by it directly rather than
    // re-scanning, so a guard call with an unbalanced arg list is caught.
    let guard = GUARD_LOOKING_OPS
      .iter()
      .copied()
      .find(|&g| g == op.as_str());
    return ParseOutcome::Unparsed { guard };
  };
  let args = rest[open + 1..close].to_string();
  let attrs = rest[close + 1..].trim().to_string();

  let const_val = (op == "const").then(|| const_scalar(&attrs)).flatten();
  ParseOutcome::Parsed {
    var: var.to_string(),
    stmt: Stmt { dtype, op, args },
    const_val,
  }
}

/// Reads a MIL program into constants, producers, and — critically — the
/// guard-looking statements it could NOT read (see [`Graph::unresolved`]).
fn parse_mil(text: &str) -> Graph {
  let mut consts = BTreeMap::new();
  let mut producers = BTreeMap::new();
  let mut unresolved = Vec::new();

  for line in text.lines() {
    let line = line.trim();
    match parse_stmt_line(line) {
      ParseOutcome::NotStatement => {}
      ParseOutcome::Parsed {
        var,
        stmt,
        const_val,
      } => {
        if let Some(value) = const_val {
          consts.insert(var.clone(), value);
        }
        producers.insert(var, stmt);
      }
      ParseOutcome::Unparsed { guard: Some(op) } => {
        unresolved.push(format!("unreadable `{op}` statement: {line}"));
      }
      ParseOutcome::Unparsed { guard: None } => {}
    }
  }

  Graph {
    consts,
    producers,
    unresolved,
  }
}

/// Extracts a scalar `const`'s value from its attribute list:
/// `[name = .., val = tensor<fp32, []>(0x1p-149)]`. Non-scalar constants
/// (weights, shapes) have a non-empty shape and are deliberately ignored.
fn const_scalar(attrs: &str) -> Option<f64> {
  let val = attrs.find("val")?;
  let open = attrs[val..].find("(")? + val;
  // Only `tensor<TY, []>` — a scalar — qualifies.
  if !attrs[val..open].replace(' ', "").contains(",[]>") {
    return None;
  }
  let close = attrs[open..].find(')')? + open;
  parse_scalar(&attrs[open + 1..close])
}

impl Graph {
  /// Resolves a token to a constant value, if it is one.
  fn value(&self, tok: Option<&str>) -> Option<f64> {
    let tok = tok?;
    parse_scalar(tok).or_else(|| self.consts.get(tok).copied())
  }

  /// Resolves a token to a constant scalar, following `cast` producers. A
  /// fp16 conversion routinely emits a guard constant as `const → cast` — an
  /// fp32 literal cast to fp16 before it reaches an `add`/`maximum`/`clip`
  /// guard operand — and the bare [`Graph::value`] stops at the `cast`, so the
  /// guard's floor silently vanishes (the exact `const(1e-8) → cast →
  /// add(count, eps) → real_div` hole a re-conversion can open). Tries the
  /// direct literal/const first — identical to [`Graph::value`] on today's
  /// tree, where every guard constant is a direct literal — then follows a
  /// `cast` chain to its constant source. Depth-capped like [`Graph::floor`].
  fn const_through_cast(&self, tok: Option<&str>, depth: u8) -> Option<f64> {
    let tok = tok?;
    if let Some(v) = self.value(Some(tok)) {
      return Some(v);
    }
    if depth > 6 {
      return None;
    }
    match self.producers.get(tok) {
      Some(stmt) if stmt.op == "cast" => self.const_through_cast(arg(&stmt.args, "x"), depth + 1),
      _ => None,
    }
  }

  /// Follows `cast` producers to the ultimate non-`cast` producing statement of
  /// `tok`, with the SAME bounded traversal [`Graph::floor`] and
  /// [`Graph::const_through_cast`] use (depth-capped at 6). A fp16 conversion
  /// routinely interposes a `cast` between a floor-contributing guard op and the
  /// site it guards (`add → cast → real_div`), so inspecting only the immediate
  /// producer would miss the guard.
  fn producer_through_cast(&self, tok: Option<&str>, depth: u8) -> Option<&Stmt> {
    let tok = tok?;
    if depth > 6 {
      return None;
    }
    match self.producers.get(tok) {
      Some(stmt) if stmt.op == "cast" => {
        self.producer_through_cast(arg(&stmt.args, "x"), depth + 1)
      }
      other => other,
    }
  }

  /// True when `operand`'s producer — followed through any `cast` chain — is a
  /// floor-contributing GUARD op (`add`/`maximum`/`clip`) whose floor
  /// nonetheless did NOT resolve: the graph structurally intends a floor here
  /// but its constant is unreadable even through casts, so it is a hole to
  /// surface, not "no claim". The `cast` chain is followed with the SAME bounded
  /// traversal [`Graph::floor`] uses (shared [`Graph::producer_through_cast`]);
  /// [`Graph::floor`] itself recursively unwraps `cast`, so a `dynamic → add →
  /// cast → real_div` divisor reaches the `add` when the floor is resolved — the
  /// unresolved check must reach it too, or the site contributes neither a
  /// finding nor a hole and simply vanishes. Kept distinct from a genuinely
  /// dynamic input (no producer, or a non-guard producer like
  /// `real_div`/`sqrt`/`mul`/`reduce_*`/`scatter`), which stays silent: that is
  /// what lets the shipped embedders' `sqrt(real_div(..))` std sites and every
  /// `x / <dynamic>` divide avoid flooding the sweep with false holes while a
  /// `count + <dynamic>` guard the reader cannot read is still caught (see the
  /// `sqrt`/`real_div` arms of [`Graph::audit`]).
  fn unreadable_floor_guard(&self, operand: Option<&str>) -> bool {
    self
      .producer_through_cast(operand, 0)
      .is_some_and(|stmt| matches!(stmt.op.as_str(), "add" | "maximum" | "clip"))
  }

  /// The provable lower bound on the tensor `var`, and where it comes from.
  ///
  /// Returns `None` when nothing constant bounds it — a dynamic value this
  /// gate deliberately makes no claim about, rather than guessing.
  fn floor(&self, var: Option<&str>, depth: u8) -> Option<(f64, String)> {
    let var = var?;
    if depth > 6 {
      return None;
    }
    let stmt = self.producers.get(var)?;
    match stmt.op.as_str() {
      "const" => self.consts.get(var).map(|v| (*v, format!("const({v:e})"))),
      // `x + eps` — the classic explicit guard. The constant operand is
      // resolved through any `cast` (a fp16 conversion casts an fp32 literal
      // before adding it), not just direct literals/consts — see
      // [`Graph::const_through_cast`].
      "add" => ["y", "x"]
        .iter()
        .find_map(|k| self.const_through_cast(arg(&stmt.args, k), 0))
        .map(|c| (c, format!("add(+{c:e})"))),
      "clip" => self
        .const_through_cast(arg(&stmt.args, "alpha"), 0)
        .map(|lo| (lo, format!("clip(alpha={lo:e})"))),
      "maximum" => ["y", "x"]
        .iter()
        .find_map(|k| self.const_through_cast(arg(&stmt.args, k), 0))
        .map(|c| (c, format!("maximum({c:e})"))),
      // A softmax output can underflow to exactly 0 in fp16 long before
      // the log's epsilon is ever added — the decomposed-log_softmax trap.
      "softmax" => Some((0.0, "softmax->log".to_string())),
      "cast" => self.floor(arg(&stmt.args, "x"), depth + 1),
      _ => None,
    }
  }

  /// Every guard site in the graph, in a stable order, together with every
  /// guard-looking statement the audit could not fully resolve. A non-empty
  /// [`Audit::unresolved`] is a hard sweep failure: an epsilon this reader
  /// cannot resolve is a hole (the guard is unreadable), never a silent pass.
  fn audit(&self) -> Audit {
    let mut found = Vec::new();
    // Parser-level holes (unreadable statement shapes) carry through; audit-
    // level holes (a recognized site whose epsilon will not resolve) join
    // them below.
    let mut unresolved = self.unresolved.clone();
    for (var, stmt) in &self.producers {
      let eps_kwarg = self.value(arg(&stmt.args, "epsilon"));
      let site = match stmt.op.as_str() {
        // `log` and `rsqrt` always carry an `epsilon` in CoreML MIL, so they
        // are always guard sites — including when that epsilon has already
        // been folded to a literal `0x0p+0`. An epsilon that will NOT resolve
        // is a hole (we cannot read the guard), not a site to fold to zero.
        "log" | "rsqrt" => match eps_kwarg {
          Some(eps) => {
            let (floor, guard) = self
              .floor(arg(&stmt.args, "x"), 0)
              .unwrap_or((0.0, "-".into()));
            Some((eps, floor, guard))
          }
          None => {
            unresolved.push(unresolved_site(var, stmt));
            None
          }
        },
        // A normalization's epsilon is its whole guard. Unresolvable means the
        // guard is unreadable, not absent — the site must FAIL the audit, not
        // vanish from it (the old `eps_kwarg.map(...)` dropped it silently).
        "instance_norm" | "layer_norm" | "batch_norm" => match eps_kwarg {
          Some(e) => Some((e, 0.0, "norm".into())),
          None => {
            unresolved.push(unresolved_site(var, stmt));
            None
          }
        },
        // `sqrt` has no epsilon: it is a guard site only when something
        // constant floors its input. A genuinely dynamic input is no claim,
        // not a hole — its guard, if any, lives in a floor-contributing op
        // whose own unreadability is caught at parse time. But a
        // floor-contributing GUARD op (`add`/`maximum`/`clip`) whose constant
        // will NOT resolve — even through a `cast` — IS a hole: the graph
        // structurally intends a floor here and the reader cannot read it, so
        // it must FAIL the audit, not `.map`-drop into silence (the
        // `const(1e-8) → cast → add → sqrt` shape a re-conversion can emit).
        "sqrt" => match self.floor(arg(&stmt.args, "x"), 0) {
          Some((f, g)) => Some((0.0, f, g)),
          None => {
            if self.unreadable_floor_guard(arg(&stmt.args, "x")) {
              unresolved.push(unresolved_site(var, stmt));
            }
            None
          }
        },
        // A divide is a guard site when its DIVISOR is const-floored —
        // the `x / (n + eps)` pooling shape. An unreadable floor guard on the
        // divisor is a hole for the same reason as `sqrt` above; a divisor with
        // no readable floor and no guard-op producer is a genuinely dynamic
        // divide, which stays "no claim".
        "real_div" => match self.floor(arg(&stmt.args, "y"), 0) {
          Some((f, g)) => Some((0.0, f, format!("denom:{g}"))),
          None => {
            if self.unreadable_floor_guard(arg(&stmt.args, "y")) {
              unresolved.push(unresolved_site(var, stmt));
            }
            None
          }
        },
        _ => None,
      };
      if let Some((eps, floor, guard)) = site {
        found.push(Finding {
          op: stmt.op.clone(),
          dtype: stmt.dtype.clone(),
          var: var.clone(),
          eps,
          floor,
          guard,
        });
      }
    }
    Audit {
      findings: found,
      unresolved,
    }
  }
}

/// The result of auditing a graph: every resolved guard site, plus every
/// guard-looking statement that could not be fully read. Completeness lives
/// here — a non-empty `unresolved` is a hard sweep failure, so a partial
/// parse can never report a clean fp16 sweep.
struct Audit {
  findings: Vec<Finding>,
  unresolved: Vec<String>,
}

/// A one-line completeness failure for a recognized guard site whose epsilon
/// did not resolve to a constant — the statement quoted so the hole is
/// actionable (which op, which var, and the arguments as read).
fn unresolved_site(var: &str, stmt: &Stmt) -> String {
  format!(
    "unresolvable epsilon on {}/{} {var}: {}({})",
    stmt.op, stmt.dtype, stmt.op, stmt.args
  )
}

/// One numerically-guarded op, with the guard resolved to a number.
struct Finding {
  op: String,
  dtype: String,
  var: String,
  /// The op's own `epsilon =` argument (0.0 when it has none).
  eps: f64,
  /// The provable lower bound on the guarded input / divisor.
  floor: f64,
  guard: String,
}

impl Finding {
  /// What the guard is actually worth: `log(x + eps)` is safe iff
  /// `eps + lower_bound(x)` is representable in fp16.
  fn effective(&self) -> f64 {
    self.eps + self.floor
  }

  /// The gate. An epsilon at or above fp16's smallest subnormal survives
  /// the conversion; anything below it rounds to zero and goes inert.
  fn survives_fp16(&self) -> bool {
    self.effective() >= FP16_MIN_SUBNORMAL
  }

  /// A `softmax` feeding a `log` is a decomposed `log_softmax`. Even with
  /// a surviving epsilon it is lossy: the softmax output underflows to 0
  /// in fp16 *before* the log ever adds the epsilon, so the true log-prob
  /// is clamped at `log(eps)` instead of computed. A fused, stable
  /// `log_softmax` (`x - logsumexp(x)`) never materializes the underflow.
  fn is_decomposed_log_softmax(&self) -> bool {
    self.op == "log" && self.guard == "softmax->log"
  }

  /// Stable one-line rendering — this is what [`KNOWN_DEFECTS`] pins, so
  /// any drift in op, dtype, guard shape, or epsilon fails the gate.
  fn render(&self) -> String {
    format!(
      "{}/{} guard={} eff={:e}",
      self.op,
      self.dtype,
      self.guard,
      self.effective()
    )
  }
}

// ---------------------------------------------------------------------------
// The pins
// ---------------------------------------------------------------------------

/// A model we knowingly still ship with an inert fp16 guard.
struct KnownDefect {
  /// Path relative to `Models/`.
  path: &'static str,
  /// Every vanishing guard site, rendered by [`Finding::render`], sorted.
  ///
  /// Pinned in BOTH directions on purpose: a new vanishing site fails the
  /// gate, and so does a *repair*. A model quietly re-converted with a
  /// healthy epsilon must not slip by unnoticed — the fix has to be seen,
  /// the pin deleted, and the parity goldens re-cut deliberately.
  sites: &'static [&'static str],
  /// What breaks, and why it is still here.
  note: &'static str,
}

/// Every fp16-vanishing guard in the tree, as of the sweep that created
/// this gate. Each entry is a defect, not an exemption.
const KNOWN_DEFECTS: &[KnownDefect] = &[
  KnownDefect {
    path: "alignkit/base960h_aligner.mlmodelc",
    sites: &["log/fp16 guard=softmax->log eff=1.401298464324817e-45"],
    note: "Decomposed log-softmax; eps 0x1p-149 rounds to 0 in fp16. `emissions` IS the log \
           output, so ANE log(0) -> ~-45440 lands directly in the shipped tensor: 16.7% of \
           output cells corrupted, word timings shifted up to 881 ms.",
  },
  KnownDefect {
    path: "speakerkit/pyannote_segmentation.mlmodelc",
    sites: &["log/fp16 guard=softmax->log eff=0e0"],
    note: "The fp16 sibling of Segmentation.mlmodelc: coremltools already folded the epsilon to \
           a literal 0x0p+0 at conversion. `segments` IS the log output. Worst logit delta \
           45,422; the shipping diarizer returns 5 of 8 speakers at 16.6% DER where the ONNX \
           reference is frame-perfect.",
  },
  KnownDefect {
    path: "speakerkit/Segmentation.mlmodelc",
    sites: &["log/fp32 guard=softmax->log eff=1.401298464324817e-45"],
    note: "Same source model and same coremltools default epsilon as pyannote_segmentation; the \
           fp32 artifact merely KEEPS 0x1p-149 rather than folding it to zero. That is fp32's \
           smallest subnormal — it survives fp32 arithmetic and nothing else. Loaded under the \
           default ComputeUnits::All it is demoted to fp16 on the ANE and vanishes exactly like \
           its fp16 sibling.",
  },
  KnownDefect {
    path: "speakerkit/wespeaker.mlmodelc",
    // THREE identical divisor guards, one per attentive-stat pooling division
    // (the weighted mean and the two divisions feeding the weighted variance /
    // `std`). Listed thrice, not deduped: the multiplicity is the blast radius
    // (finding 5).
    sites: &[
      "real_div/fp32 guard=denom:add(+9.99999993922529e-9) eff=9.99999993922529e-9",
      "real_div/fp32 guard=denom:add(+9.99999993922529e-9) eff=9.99999993922529e-9",
      "real_div/fp32 guard=denom:add(+9.99999993922529e-9) eff=9.99999993922529e-9",
    ],
    note: "Attentive-stat pooling divides by `count + 1e-8` at THREE sites (the weighted mean \
           and the two divisions behind the weighted variance/std). 1e-8 is 0.168x fp16's \
           smallest subnormal, so on the ANE all three divisor guards are zero. Same fp32 \
           artifact, same input, only CpuOnly -> All: cosine collapses to 0.035.",
  },
  KnownDefect {
    path: "speakerkit/wespeaker_v2.mlmodelc",
    sites: &[
      "real_div/fp32 guard=denom:add(+9.99999993922529e-9) eff=9.99999993922529e-9",
      "real_div/fp32 guard=denom:add(+9.99999993922529e-9) eff=9.99999993922529e-9",
      "real_div/fp32 guard=denom:add(+9.99999993922529e-9) eff=9.99999993922529e-9",
    ],
    note: "Same three-site pooling epsilon as wespeaker.mlmodelc.",
  },
  KnownDefect {
    path: "speakerkit/wespeaker_int8.mlmodelc",
    sites: &[
      "real_div/fp32 guard=denom:add(+9.99999993922529e-9) eff=9.99999993922529e-9",
      "real_div/fp32 guard=denom:add(+9.99999993922529e-9) eff=9.99999993922529e-9",
      "real_div/fp32 guard=denom:add(+9.99999993922529e-9) eff=9.99999993922529e-9",
    ],
    note: "Same three-site pooling epsilon as wespeaker.mlmodelc.",
  },
  KnownDefect {
    path: "speakerkit/PLDA.mlmodelc",
    // TWO sqrt-of-clipped-value guards, each clipped to 1e-12 (finding 5).
    sites: &[
      "sqrt/fp32 guard=clip(alpha=9.999999960041972e-13) eff=9.999999960041972e-13",
      "sqrt/fp32 guard=clip(alpha=9.999999960041972e-13) eff=9.999999960041972e-13",
    ],
    note: "Normalization clips to 1e-12 before `sqrt` at TWO sites, then divides by it. 1e-12 is \
           1.7e-5x fp16's smallest subnormal: on the ANE the clip floor is zero, giving sqrt(0) \
           and a divide by zero. Not yet observed in a shipping path (found by this sweep, not by \
           a failure).",
  },
  KnownDefect {
    path: "speakerkit/PldaRho.mlmodelc",
    sites: &[
      "sqrt/fp32 guard=clip(alpha=9.999999960041972e-13) eff=9.999999960041972e-13",
      "sqrt/fp32 guard=clip(alpha=9.999999960041972e-13) eff=9.999999960041972e-13",
    ],
    note: "Same two-site 1e-12 clip floor as PLDA.mlmodelc.",
  },
  KnownDefect {
    path: "argmax-speakerkit/speaker_segmenter/pyannote-v3/W32A32/SpeakerSegmenter.mlmodelc",
    sites: &["log/fp16 guard=softmax->log eff=0e0"],
    note: "Vendored from argmax. Epsilon already folded to 0x0p+0, and the graph is fp16 \
           DESPITE the W32A32 directory name. Contained, not silent-clean: the saturated log \
           feeds an `exp` that maps it back toward 0 before any shipped output, and the winning \
           powerset class never underflows, so `speaker_probs`/`speaker_ids` survive. The guard \
           is still inert — pinned so a re-vendored graph cannot widen the blast radius unseen.",
  },
  KnownDefect {
    path: "argmax-speakerkit/speaker_segmenter/pyannote-v3/W8A16/SpeakerSegmenter.mlmodelc",
    sites: &["log/fp16 guard=softmax->log eff=1.401298464324817e-45"],
    note: "Same graph as the W32A32 variant with the epsilon left at 0x1p-149 instead of folded \
           to zero — identically inert in fp16, identically contained by the downstream `exp`.",
  },
];

// ---------------------------------------------------------------------------
// Hermetic parser tests — no models, always run.
//
// These are the gate's can-it-fail proof, kept permanently executable:
// every snippet below is a VERBATIM excerpt of a real shipped `model.mil`.
// ---------------------------------------------------------------------------

/// `Models/alignkit/base960h_aligner.mlmodelc/model.mil`, lines 800-803.
const ALIGNKIT_LOG_SOFTMAX: &str = r#"
            tensor<int32, []> var_847 = const()[name = tensor<string, []>("op_847"), val = tensor<int32, []>(-1)];
            tensor<fp16, [1, 2999, 29]> var_849_softmax_cast_fp16 = softmax(axis = var_847, x = linear_73_cast_fp16)[name = tensor<string, []>("op_849_softmax_cast_fp16")];
            tensor<fp32, []> var_849_epsilon_0 = const()[name = tensor<string, []>("op_849_epsilon_0"), val = tensor<fp32, []>(0x1p-149)];
            tensor<fp16, [1, 2999, 29]> var_849_cast_fp16 = log(epsilon = var_849_epsilon_0, x = var_849_softmax_cast_fp16)[name = tensor<string, []>("op_849_cast_fp16")];
"#;

/// `Models/speakerkit/pyannote_segmentation.mlmodelc/model.mil`, lines 137-139.
const SPEAKERKIT_SEG_FP16: &str = r#"
            tensor<fp16, [1, 589, 7]> var_231_softmax_cast_fp16 = softmax(axis = var_230, x = linear_2_cast_fp16)[name = tensor<string, []>("op_231_softmax_cast_fp16")];
            tensor<fp16, []> var_231_epsilon_0_to_fp16 = const()[name = tensor<string, []>("op_231_epsilon_0_to_fp16"), val = tensor<fp16, []>(0x0p+0)];
            tensor<fp16, [1, 589, 7]> var_231_cast_fp16 = log(epsilon = var_231_epsilon_0_to_fp16, x = var_231_softmax_cast_fp16)[name = tensor<string, []>("op_231_cast_fp16")];
"#;

/// `Models/speakerkit/wespeaker.mlmodelc/model.mil`, lines 4444-4450.
const WESPEAKER_POOLING: &str = r#"
            tensor<fp32, []> var_5790 = const()[name = tensor<string, []>("op_5790"), val = tensor<fp32, []>(0x1.5798eep-27)];
            tensor<fp32, [3, 1]> v1 = add(x = var_5789, y = var_5790)[name = tensor<string, []>("v1")];
            tensor<fp32, [3, 2560]> mean = real_div(x = var_5794, y = v1)[name = tensor<string, []>("mean")];
"#;

/// `Models/whisperkit-coreml/openai_whisper-tiny/MelSpectrogram.mlmodelc/model.mil`,
/// lines 46-49. The one graph in the workspace that got this right — and
/// note it did so with an explicit `add`, not with the `log`'s own
/// epsilon, which is a literal `0x0p+0` here too.
const WHISPER_MEL: &str = r#"
            tensor<fp16, []> var_41_to_fp16 = const()[name = tensor<string, []>("op_41_to_fp16"), val = tensor<fp16, []>(0x1p-24)];
            tensor<fp16, [80, 3000]> mel_spec_cast_fp16 = add(x = mel_spec_1_cast_fp16, y = var_41_to_fp16)[name = tensor<string, []>("mel_spec_cast_fp16")];
            tensor<fp16, []> log_0_epsilon_0_to_fp16 = const()[name = tensor<string, []>("log_0_epsilon_0_to_fp16"), val = tensor<fp16, []>(0x0p+0)];
            tensor<fp16, [80, 3000]> log_0_cast_fp16 = log(epsilon = log_0_epsilon_0_to_fp16, x = mel_spec_cast_fp16)[name = tensor<string, []>("log_0_cast_fp16")];
"#;

/// Two INDEPENDENT decomposed-`log_softmax` sites, distinct vars but an
/// identical vanishing signature — the multiset shape finding 5 is about, and
/// the exact one the live wespeaker/PLDA graphs carry (3 and 2 same-signature
/// sites). Both render to `log/fp16 guard=softmax->log eff=0e0`, so a `dedup` or
/// set would collapse them into one and hide the second defect under a green
/// pin. Synthesized from two copies of the real `SPEAKERKIT_SEG_FP16` shape.
const TWO_SITE_LOG_SOFTMAX: &str = r#"
            tensor<fp16, [1, 589, 7]> a_softmax_cast_fp16 = softmax(axis = a_axis, x = a_linear)[name = tensor<string, []>("a_softmax")];
            tensor<fp16, []> a_epsilon = const()[name = tensor<string, []>("a_epsilon"), val = tensor<fp16, []>(0x0p+0)];
            tensor<fp16, [1, 589, 7]> a_cast_fp16 = log(epsilon = a_epsilon, x = a_softmax_cast_fp16)[name = tensor<string, []>("a_log")];
            tensor<fp16, [1, 589, 7]> b_softmax_cast_fp16 = softmax(axis = b_axis, x = b_linear)[name = tensor<string, []>("b_softmax")];
            tensor<fp16, []> b_epsilon = const()[name = tensor<string, []>("b_epsilon"), val = tensor<fp16, []>(0x0p+0)];
            tensor<fp16, [1, 589, 7]> b_cast_fp16 = log(epsilon = b_epsilon, x = b_softmax_cast_fp16)[name = tensor<string, []>("b_log")];
"#;

/// One clean, surviving guard (whisper's mel `add(0x1p-24) -> log`) beside a
/// second `log` emitted in syntax this reader cannot parse — its argument
/// list left unbalanced, a stand-in for the unhandled shapes a new coremltools
/// re-conversion can produce. This is the partial-parse trap: the recognized
/// guard alone must NOT let the sweep report success while the unreadable
/// vanishing guard silently disappears. The audit is required to surface the
/// second statement as unresolved, not drop it.
const VALID_GUARD_PLUS_UNREADABLE_GUARD: &str = r#"
            tensor<fp16, []> ok_eps = const()[name = tensor<string, []>("ok_eps"), val = tensor<fp16, []>(0x1p-24)];
            tensor<fp16, [80, 3000]> ok_mel = add(x = ok_mel_1, y = ok_eps)[name = tensor<string, []>("ok_mel")];
            tensor<fp16, []> ok_log_eps = const()[name = tensor<string, []>("ok_log_eps"), val = tensor<fp16, []>(0x0p+0)];
            tensor<fp16, [80, 3000]> ok_log = log(epsilon = ok_log_eps, x = ok_mel)[name = tensor<string, []>("ok_log")];
            tensor<fp16, [1, 589, 7]> bad_softmax = softmax(axis = bad_axis, x = bad_linear)[name = tensor<string, []>("bad_softmax")];
            tensor<fp16, [1, 589, 7]> bad_log = log(epsilon = bad_eps, x = bad_softmax [name = tensor<string, []>("bad_log")];
"#;

/// A `batch_norm` whose `epsilon` names a var that is never defined as a
/// scalar const — the guard is present but unreadable. The old
/// `eps_kwarg.map(...)` dropped such a site silently (the `.map` short-circuits
/// on `None`); completeness requires it to FAIL the audit with the statement
/// quoted, exactly as a malformed parse does.
const NORM_WITH_UNRESOLVABLE_EPSILON: &str = r#"
            tensor<fp16, [1, 384, 1, 1500]> n_out = batch_norm(beta = n_beta, epsilon = n_eps_missing, gamma = n_gamma, mean = n_mean, variance = n_var, x = n_in)[name = tensor<string, []>("n_out")];
"#;

/// A pooling-divisor guard emitted as `const → cast → add → real_div` — the
/// shape a coremltools re-conversion produces when it casts the fp32 epsilon
/// literal to fp16 before adding it to the count. Every statement here parses,
/// so NOTHING is unresolved at parse time; the `1e-8` floor is reachable ONLY
/// by following the `cast` from the `add`'s operand to the const. Modeled on
/// the real `WESPEAKER_POOLING` `count + 1e-8` guard with a `cast` interposed
/// on the constant. Before the audit followed constants through `cast`,
/// `floor(add)` missed the cast-wrapped const, the `real_div` `.map`-dropped to
/// nothing, and this vanishing guard produced NEITHER a finding NOR an
/// unresolved hole — it simply disappeared while any other recognized guard
/// kept the sweep GREEN.
const CAST_WRAPPED_POOLING_DIVISOR: &str = r#"
            tensor<fp32, []> eps_fp32 = const()[name = tensor<string, []>("eps_fp32"), val = tensor<fp32, []>(0x1.5798eep-27)];
            tensor<fp16, []> eps_fp16 = cast(dtype = fp16, x = eps_fp32)[name = tensor<string, []>("eps_fp16")];
            tensor<fp16, [3, 1]> v1 = add(x = count_cast_fp16, y = eps_fp16)[name = tensor<string, []>("v1")];
            tensor<fp16, [3, 2560]> mean = real_div(x = numer_cast_fp16, y = v1)[name = tensor<string, []>("mean")];
"#;

/// The same pooling-divisor shape, but the epsilon is DYNAMIC — computed (a
/// `mul`), not a constant — so no `cast` chain reaches a literal. The `add`
/// still structurally guards the divisor (`count + <something>`), so its
/// unresolvable floor is a HOLE, not "no claim": the reader can see a guard it
/// cannot read, and must surface the `real_div` as unresolved rather than
/// `.map`-drop it into a silent pass. Contrast a genuinely dynamic divisor
/// produced by a NON-guard op — the shipped embedders' `x / real_div(..)` and
/// `sqrt(real_div(..))` std sites — which stays "no claim" and never lands
/// here.
const DYNAMIC_UNRESOLVABLE_DIVISOR: &str = r#"
            tensor<fp16, [3, 1]> dyn_eps = mul(x = a_cast_fp16, y = b_cast_fp16)[name = tensor<string, []>("dyn_eps")];
            tensor<fp16, [3, 1]> v1 = add(x = count_cast_fp16, y = dyn_eps)[name = tensor<string, []>("v1")];
            tensor<fp16, [3, 2560]> mean = real_div(x = numer_cast_fp16, y = v1)[name = tensor<string, []>("mean")];
"#;

/// The same dynamically-unresolvable `add`-guarded divisor, but with a `cast`
/// interposed before the `real_div` (`mul(dynamic) → add → cast → real_div`),
/// beside one clean, surviving guard (whisper's mel `add(0x1p-24) → log`). Every
/// statement parses, so nothing is unresolved at parse time; the `add`
/// structurally guards the divisor but its epsilon is a `mul` (dynamic), so no
/// floor resolves. `Graph::floor` recursively unwraps the `cast` to reach the
/// `add` — so the unresolved-detection MUST unwrap it too, or the site produces
/// NEITHER a finding NOR a hole and simply disappears while the clean guard
/// keeps the sweep green. This is the exact hole an `unreadable_floor_guard`
/// that inspects only the divisor's IMMEDIATE producer (the `cast`, whose op is
/// not a guard op) leaves open.
const CAST_WRAPPED_DYNAMIC_DIVISOR: &str = r#"
            tensor<fp16, []> ok_eps = const()[name = tensor<string, []>("ok_eps"), val = tensor<fp16, []>(0x1p-24)];
            tensor<fp16, [80, 3000]> ok_mel = add(x = ok_mel_1, y = ok_eps)[name = tensor<string, []>("ok_mel")];
            tensor<fp16, []> ok_log_eps = const()[name = tensor<string, []>("ok_log_eps"), val = tensor<fp16, []>(0x0p+0)];
            tensor<fp16, [80, 3000]> ok_log = log(epsilon = ok_log_eps, x = ok_mel)[name = tensor<string, []>("ok_log")];
            tensor<fp16, [3, 1]> dyn_eps = mul(x = a_cast_fp16, y = b_cast_fp16)[name = tensor<string, []>("dyn_eps")];
            tensor<fp16, [3, 1]> v1 = add(x = count_cast_fp16, y = dyn_eps)[name = tensor<string, []>("v1")];
            tensor<fp16, [3, 1]> v1_fp16 = cast(dtype = fp16, x = v1)[name = tensor<string, []>("v1_fp16")];
            tensor<fp16, [3, 2560]> mean = real_div(x = numer_cast_fp16, y = v1_fp16)[name = tensor<string, []>("mean")];
"#;

/// The vanishing guard sites of an already-audited graph, rendered and sorted
/// as a **multiset** — duplicates PRESERVED. Two sites with the same signature
/// are two defects, not one: a `dedup`/set here would let a reconversion that
/// adds a second same-signature vanishing site collapse into the first and keep
/// a green pin while the blast radius grows (finding 5). The live tree already
/// carries this — wespeaker's attentive-stat pooling has THREE identical
/// `real_div` guards and PLDA/PldaRho two `sqrt` each — which is exactly why the
/// dedup was hiding real multiplicity.
///
/// Both the sweep and the hermetic parser tests route through here, so the
/// multiset property is exercised by the always-run tests, not merely asserted
/// against live models.
fn vanishing_sites(findings: &[Finding]) -> Vec<String> {
  let mut sites: Vec<String> = findings
    .iter()
    .filter(|f| !f.survives_fp16())
    .map(Finding::render)
    .collect();
  sites.sort();
  sites
}

/// The vanishing guard sites of a MIL program (parse + audit + multiset render).
/// Only `log`/`sqrt`/`rsqrt`/`real_div`/norm sites, not every op.
///
/// Panics if the audit left any guard-looking statement unresolved, so every
/// hermetic snippet that routes through here doubles as proof it parsed
/// completely — a dropped guard can never hide inside a merely-empty vanishing
/// list.
fn vanishing(mil: &str) -> Vec<String> {
  let audit = parse_mil(mil).audit();
  assert!(
    audit.unresolved.is_empty(),
    "audit left guard-looking statement(s) unresolved: {:?}",
    audit.unresolved
  );
  vanishing_sites(&audit.findings)
}

#[test]
fn threshold_is_fp16s_smallest_subnormal() {
  assert_eq!(
    FP16_MIN_SUBNORMAL,
    2.0_f64.powi(-24),
    "the gate's threshold must be exactly 2^-24"
  );
  assert_eq!(FP16_MIN_NORMAL, 2.0_f64.powi(-14));

  // Corroborate the threshold against a real fp16 rounding, so the
  // constant above cannot drift away from the format it claims to model.
  assert_eq!(
    half::f16::from_f64(2.0_f64.powi(-149)),
    half::f16::from_f64(0.0),
    "0x1p-149 must round to zero in fp16"
  );
  assert_eq!(half::f16::from_f64(1e-8), half::f16::from_f64(0.0));
  assert_eq!(half::f16::from_f64(1e-12), half::f16::from_f64(0.0));
  assert!(
    half::f16::from_f64(FP16_MIN_SUBNORMAL) > half::f16::from_f64(0.0),
    "2^-24 must be representable in fp16"
  );
}

#[test]
fn hex_float_literals_parse_exactly() {
  assert_eq!(parse_hex_float("0x1p-149"), Some(2.0_f64.powi(-149)));
  assert_eq!(parse_hex_float("0x0p+0"), Some(0.0));
  assert_eq!(parse_hex_float("0x1p-24"), Some(FP16_MIN_SUBNORMAL));
  // 1e-8 and 1e-12, as coremltools actually spells them.
  let eight = parse_hex_float("0x1.5798eep-27").expect("parses");
  assert!(
    (eight - 1e-8).abs() < 1e-15,
    "0x1.5798eep-27 ~= 1e-8, got {eight:e}"
  );
  let twelve = parse_hex_float("0x1.197998p-40").expect("parses");
  assert!(
    (twelve - 1e-12).abs() < 1e-19,
    "0x1.197998p-40 ~= 1e-12, got {twelve:e}"
  );
}

/// The gate must FAIL on the real alignkit graph. If this ever passes, the
/// checker has stopped checking.
#[test]
fn detects_the_alignkit_log_softmax_defect() {
  assert_eq!(
    vanishing(ALIGNKIT_LOG_SOFTMAX),
    ["log/fp16 guard=softmax->log eff=1.401298464324817e-45"],
    "alignkit's fp16 log(eps = 0x1p-149) must be caught"
  );

  let graph = parse_mil(ALIGNKIT_LOG_SOFTMAX);
  let audit = graph.audit();
  assert!(
    audit.unresolved.is_empty(),
    "the real alignkit excerpt must parse completely: {:?}",
    audit.unresolved
  );
  let log = audit
    .findings
    .iter()
    .find(|f| f.op == "log")
    .expect("a log site");
  assert!(!log.survives_fp16());
  assert!(
    log.is_decomposed_log_softmax(),
    "and it must be recognized as a decomposed log_softmax"
  );
}

/// The other face of the same defect: a divisor guard, not a log epsilon.
#[test]
fn detects_the_wespeaker_pooling_defect() {
  assert_eq!(
    vanishing(WESPEAKER_POOLING),
    ["real_div/fp32 guard=denom:add(+9.99999993922529e-9) eff=9.99999993922529e-9"],
    "wespeaker's `count + 1e-8` divisor guard must be caught even though \
     the graph declares fp32 — the ANE demotes it to fp16 regardless"
  );
}

/// And the already-folded-to-zero face.
#[test]
fn detects_the_speakerkit_segmentation_defect() {
  assert_eq!(
    vanishing(SPEAKERKIT_SEG_FP16),
    ["log/fp16 guard=softmax->log eff=0e0"],
    "an epsilon coremltools already folded to 0x0p+0 must be caught"
  );
}

/// The control. whisperkit's mel guards its `log` with an explicit
/// `add(x, 0x1p-24)` — exactly fp16's smallest subnormal — so it survives,
/// and the gate must say so. A checker that flagged this too would be
/// useless (everything fails, nobody looks).
#[test]
fn accepts_whisperkits_mel_guard() {
  assert_eq!(
    vanishing(WHISPER_MEL),
    Vec::<String>::new(),
    "whisper's mel add(x, 0x1p-24) is exactly at the fp16 floor and survives"
  );

  let graph = parse_mil(WHISPER_MEL);
  let audit = graph.audit();
  assert!(
    audit.unresolved.is_empty(),
    "the real whisper-mel excerpt must parse completely: {:?}",
    audit.unresolved
  );
  let log = audit
    .findings
    .iter()
    .find(|f| f.op == "log")
    .expect("a log site");
  assert_eq!(log.eps, 0.0, "the log's OWN epsilon is 0x0p+0 here");
  assert_eq!(
    log.floor, FP16_MIN_SUBNORMAL,
    "the guard is the preceding add, not the log's epsilon"
  );
  assert!(log.survives_fp16());
  assert!(
    !log.is_decomposed_log_softmax(),
    "it logs a mel spectrogram, not a softmax"
  );
}

/// Finding 5: two vanishing sites with the SAME signature must be TWO findings,
/// not deduped to one. This is the blast-radius multiplicity a `dedup`/set
/// silently hides (a reconversion that adds a second same-signature vanishing
/// site keeps a green pin while doubling the corrupted sites). Routes through
/// the same `vanishing` path the live sweep uses, so re-introducing a dedup in
/// [`vanishing_sites`] breaks this hermetically, before any model is loaded.
#[test]
fn same_signature_sites_are_a_multiset_not_a_set() {
  let sites = vanishing(TWO_SITE_LOG_SOFTMAX);
  assert_eq!(
    sites.len(),
    2,
    "two independent softmax->log sites must render as TWO findings, not collapse to one — got {sites:?}"
  );
  assert_eq!(
    sites[0], sites[1],
    "...and they share a signature, which is exactly what a set/dedup would fold away"
  );
  assert_eq!(sites[0], "log/fp16 guard=softmax->log eff=0e0");
}

/// Completeness (partial-parse face). A graph with ONE recognized, surviving
/// guard beside ONE guard in syntax the reader cannot parse must FAIL the
/// audit: the unreadable statement is surfaced, never dropped so the recognized
/// guard alone reports a clean sweep. This is the exact rot a re-conversion can
/// introduce — new coremltools, new MIL shape — and the reason the sweep audits
/// completeness, not merely the guards it happens to recognize. Mutating the
/// reader back to drop-silently (classifying every unparsed line as
/// non-guard) turns `unresolved` empty and this assertion red.
#[test]
fn a_partial_parse_fails_the_audit_never_reports_clean() {
  let audit = parse_mil(VALID_GUARD_PLUS_UNREADABLE_GUARD).audit();

  // The clean mel-style guard is still audited and still survives...
  assert!(
    audit
      .findings
      .iter()
      .any(|f| f.op == "log" && f.survives_fp16()),
    "the valid mel-style guard must still be audited and survive"
  );
  // ...but the unreadable second `log` is surfaced as a completeness hole,
  // not silently dropped: the audit is NOT clean, and it quotes the statement.
  assert!(
    !audit.unresolved.is_empty(),
    "an unreadable guard statement must fail the audit, not vanish — got no unresolved holes"
  );
  assert!(
    audit
      .unresolved
      .iter()
      .any(|u| u.contains("log") && u.contains("bad_log")),
    "the unresolved report must quote the offending `log` statement: {:?}",
    audit.unresolved
  );
}

/// Completeness (unresolvable-epsilon face). A recognized guard SITE
/// (`batch_norm`) whose epsilon does not resolve to a constant is a hole, not
/// an absence: the guard is unreadable. The old `eps_kwarg.map(...)`
/// short-circuited on `None` and dropped the site; it must now fail the audit
/// with the statement quoted. Reverting the norm arm to `eps_kwarg.map(...)`
/// turns `unresolved` empty and this assertion red.
#[test]
fn a_norm_with_an_unreadable_epsilon_is_a_hole_not_a_skip() {
  let audit = parse_mil(NORM_WITH_UNRESOLVABLE_EPSILON).audit();
  assert!(
    audit.findings.is_empty(),
    "an unresolvable-epsilon norm yields no resolved finding, got: {:?}",
    audit
      .findings
      .iter()
      .map(Finding::render)
      .collect::<Vec<_>>()
  );
  assert!(
    audit
      .unresolved
      .iter()
      .any(|u| u.contains("batch_norm") && u.contains("n_out")),
    "...but it must be reported unresolved, quoting the statement — not dropped: {:?}",
    audit.unresolved
  );
}

/// Regression (completeness, cast-wrapped floor). A `const → cast → add →
/// real_div` divisor guard must surface its `1e-8` site in `vanishing()`,
/// exactly like the direct-const `WESPEAKER_POOLING`. Before the audit followed
/// constants through `cast`, this chain — every statement of which parses —
/// produced no finding and no unresolved hole: the vanishing guard disappeared
/// while any other recognized guard kept the sweep green. MUTATION PROOF:
/// reverting `Graph::floor`'s `add` arm from `const_through_cast` back to
/// `value` loses the cast-wrapped floor and turns this assertion red — the
/// pinned `real_div` site is no longer what `vanishing()` returns.
#[test]
fn follows_a_cast_wrapped_pooling_divisor_guard() {
  assert_eq!(
    vanishing(CAST_WRAPPED_POOLING_DIVISOR),
    ["real_div/fp16 guard=denom:add(+9.99999993922529e-9) eff=9.99999993922529e-9"],
    "a `count + cast(1e-8)` divisor guard must be caught THROUGH the cast, \
     rendering identically to the direct-const wespeaker pooling guard"
  );
}

/// Regression (completeness, dynamically-unresolvable floor). An `add`-guarded
/// divisor whose epsilon will not resolve to a constant — even through casts —
/// is a hole the audit must FAIL on, not a silent drop. The `real_div` here
/// divides by `count + <dynamic>`: structurally a guard, unreadable in value.
/// A genuinely dynamic divisor from a NON-guard op stays silent instead (see
/// `DYNAMIC_UNRESOLVABLE_DIVISOR`). MUTATION PROOF: reverting the `real_div`
/// arm from routing this to `unresolved` back to a bare `.map`-drop turns
/// `unresolved` empty and this assertion red.
#[test]
fn an_unresolvable_add_guarded_divisor_is_a_hole_not_a_drop() {
  let audit = parse_mil(DYNAMIC_UNRESOLVABLE_DIVISOR).audit();
  assert!(
    audit.findings.is_empty(),
    "an unresolvable divisor guard yields no resolved finding, got: {:?}",
    audit
      .findings
      .iter()
      .map(Finding::render)
      .collect::<Vec<_>>()
  );
  assert!(
    audit
      .unresolved
      .iter()
      .any(|u| u.contains("real_div") && u.contains("mean")),
    "the unreadable `add`-guarded divisor must be surfaced as unresolved, quoting \
     the statement — not dropped: {:?}",
    audit.unresolved
  );
}

/// Regression (completeness, cast-wrapped UNRESOLVED floor). A cast-wrapped,
/// dynamically-unresolvable `add`-guarded divisor (`mul → add → cast →
/// real_div`) beside a clean, surviving guard must FAIL the audit: the clean
/// guard must not mask the hole. `Graph::floor` unwraps the `cast` to reach the
/// `add` when resolving the floor, so the unresolved-detection must unwrap it
/// too — otherwise the `real_div` yields neither a finding nor an unresolved
/// hole and vanishes. MUTATION PROOF: reverting `unreadable_floor_guard` to
/// inspect only the divisor's IMMEDIATE producer (dropping the shared
/// `producer_through_cast` unwrap) makes it see the `cast` — whose op is not a
/// guard op — return false, push no unresolved, and this assertion goes red (the
/// audit reports clean while the guard silently disappears).
#[test]
fn follows_a_cast_before_an_unresolvable_divisor_guard() {
  let audit = parse_mil(CAST_WRAPPED_DYNAMIC_DIVISOR).audit();

  // The clean mel-style guard is still audited and still survives...
  assert!(
    audit
      .findings
      .iter()
      .any(|f| f.op == "log" && f.survives_fp16()),
    "the valid mel-style guard must still be audited and survive"
  );
  // ...but the cast-wrapped `add`-guarded divisor is surfaced as a hole,
  // quoting the `real_div` statement — not dropped so the clean guard alone
  // reports a clean sweep.
  assert!(
    audit
      .unresolved
      .iter()
      .any(|u| u.contains("real_div") && u.contains("mean")),
    "the cast-wrapped unresolvable divisor guard must be surfaced as unresolved, \
     quoting the statement — not dropped: {:?}",
    audit.unresolved
  );
}

/// The walk propagates I/O errors instead of flattening them into a silent
/// early return: a `read_dir` failure must never quietly shrink the sweep to
/// whatever happened to be readable. The old `let Ok(entries) = .. else
/// return` swallowed exactly this.
#[test]
fn discover_propagates_read_dir_errors() {
  let mut out = Vec::new();
  let missing = models_dir().join("__no_such_subtree_for_the_walk__");
  assert!(
    discover(&missing, &mut out).is_err(),
    "read_dir on a nonexistent path must return Err, not an empty Ok"
  );
  assert!(out.is_empty(), "a failed walk collects nothing");
}

// ---------------------------------------------------------------------------
// The sweep — runs iff `Models/` is on disk (see `build.rs`).
// ---------------------------------------------------------------------------

/// Workspace `Models/`, matching the other crates' test helpers.
fn models_dir() -> PathBuf {
  PathBuf::from(env!("CARGO_MANIFEST_DIR"))
    .join("../..")
    .join("Models")
}

/// Every `*.mlmodelc` under `root`, recursively. A walk, never a list.
///
/// Dot-directories are skipped: `huggingface-cli` leaves a
/// `.cache/huggingface/download/` tree of `.mlmodelc`-NAMED bookkeeping
/// directories that hold no `model.mil` (25 of them, against 26 real
/// models). They are download metadata, not shipped artifacts — and
/// treating them as models would make the "a .mlmodelc must have a
/// readable model.mil" hard failure fire on every machine that pulled the
/// argmax models from the Hub.
///
/// A `read_dir` failure or an unreadable directory entry PROPAGATES as an
/// `Err` rather than flattening into a silent early return — the walk must
/// never quietly shrink the sweep to whatever happened to be readable, which
/// is the same "fixture went missing, test went green" mode the pins guard
/// against.
fn discover(root: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
  let entries = fs::read_dir(root)
    .map_err(|e| io::Error::new(e.kind(), format!("read_dir {}: {e}", root.display())))?;
  for entry in entries {
    let entry = entry
      .map_err(|e| io::Error::new(e.kind(), format!("dir entry under {}: {e}", root.display())))?;
    let path = entry.path();
    if !path.is_dir() {
      continue;
    }
    if entry.file_name().to_string_lossy().starts_with('.') {
      continue;
    }
    if path.extension().is_some_and(|e| e == "mlmodelc") {
      out.push(path);
    } else {
      discover(&path, out)?;
    }
  }
  Ok(())
}

#[cfg_attr(
  not(models_present),
  ignore = "Models/ is gitignored and absent — nothing to sweep (build.rs)"
)]
#[test]
fn every_shipped_model_graph_survives_fp16() {
  let root = models_dir();
  assert!(
    root.is_dir(),
    "Models/ vanished between build and run: {}",
    root.display()
  );

  let mut models = Vec::new();
  discover(&root, &mut models)
    .unwrap_or_else(|e| panic!("walking Models/ failed instead of silently skipping: {e}"));
  models.sort();

  // Non-vacuity. A sweep that found nothing must never report `ok`.
  assert!(
    !models.is_empty(),
    "Models/ exists but contains no .mlmodelc — the sweep would be vacuous"
  );

  let pins: BTreeMap<&str, &KnownDefect> = KNOWN_DEFECTS.iter().map(|d| (d.path, d)).collect();
  let mut audited_sites = 0_usize;
  let mut failures = Vec::new();
  let mut seen = Vec::new();

  for model in &models {
    let rel = model
      .strip_prefix(&root)
      .expect("discovered under root")
      .to_string_lossy()
      .replace('\\', "/");
    seen.push(rel.clone());

    // A model directory with no readable graph is a hard failure, never a skip.
    let mil = model.join("model.mil");
    let text = fs::read_to_string(&mil)
      .unwrap_or_else(|e| panic!("{rel}: .mlmodelc has no readable model.mil ({e})"));

    let Audit {
      findings,
      unresolved,
    } = parse_mil(&text).audit();

    // Completeness: a guard-looking statement the reader could not resolve is
    // a hole, not a pass. A PARTIAL parse — one recognized guard beside a new
    // vanishing one in syntax this reader does not handle — must fail the
    // sweep with the offending statement quoted, never slip through GREEN on
    // the strength of the one guard it happened to recognize.
    if !unresolved.is_empty() {
      failures.push(format!(
        "{rel}: {} guard-looking statement(s) the reader could not resolve — a partial parse \
         fails the sweep rather than dropping a guard. Re-convert with a readable guard or teach \
         the reader this shape:\n      {}",
        unresolved.len(),
        unresolved.join("\n      ")
      ));
    }

    assert!(
      !findings.is_empty(),
      "{rel}: parsed zero guard sites from a {} byte graph — the parser has rotted",
      text.len()
    );
    audited_sites += findings.len();

    // A MULTISET, not a set: duplicates are preserved so a second
    // same-signature vanishing site fails the pin instead of collapsing into the
    // first (finding 5). See `vanishing_sites`.
    let vanishing = vanishing_sites(&findings);

    // Even a SURVIVING epsilon does not make `softmax -> log` safe: the
    // softmax underflows to 0 in fp16 before the log adds it. Any such
    // composition must be pinned, whatever its epsilon.
    let decomposed: Vec<&Finding> = findings
      .iter()
      .filter(|f| f.is_decomposed_log_softmax() && f.survives_fp16())
      .collect();
    for f in decomposed {
      failures.push(format!(
        "{rel}: {} ({}) is a decomposed log_softmax. Its epsilon survives fp16, but the \
         softmax output underflows to 0 BEFORE the log adds it, clamping the true log-prob \
         at log(eps). Convert with a fused, stable log_softmax (x - logsumexp(x)).",
        f.var,
        f.render()
      ));
    }

    match pins.get(rel.as_str()) {
      Some(pin) => {
        let expected: Vec<String> = pin.sites.iter().map(|s| (*s).to_string()).collect();
        if vanishing != expected {
          if vanishing.is_empty() {
            failures.push(format!(
              "{rel}: PINNED KNOWN DEFECT IS FIXED.\n    was: {expected:?}\n    now: clean.\n    \
               If this model was re-converted, that is good news — but it must be seen: delete \
               its KNOWN_DEFECTS entry and re-cut the parity goldens deliberately.\n    Pin note: \
               {}",
              pin.note
            ));
          } else {
            failures.push(format!(
              "{rel}: pinned defect CHANGED.\n    expected: {expected:?}\n    found:    \
               {vanishing:?}\n    Pin note: {}",
              pin.note
            ));
          }
        }
      }
      None if !vanishing.is_empty() => {
        failures.push(format!(
          "{rel}: NEW fp16-vanishing guard in an unpinned model: {vanishing:?}\n    Every \
           epsilon here is below fp16's smallest subnormal ({FP16_MIN_SUBNORMAL:e}), so it \
           rounds to zero and the guard goes inert on the ANE/GPU. Re-convert with an epsilon \
           >= 2^-24, or pin it in KNOWN_DEFECTS with a note saying what breaks."
        ));
      }
      None => {}
    }
  }

  // A pinned defect that has disappeared from a vendor tree that IS
  // present means the pin can no longer be verified. Hard failure — this
  // is exactly the "fixture went missing, test went green" mode.
  for defect in KNOWN_DEFECTS {
    let vendor = defect.path.split('/').next().unwrap_or(defect.path);
    let vendor_present = root.join(vendor).is_dir();
    if vendor_present && !seen.iter().any(|s| s == defect.path) {
      failures.push(format!(
        "{}: pinned known-defect model is MISSING, but Models/{vendor}/ is present. The pin \
         cannot be verified. Restore the model or remove the pin.",
        defect.path
      ));
    }
  }

  assert!(
    audited_sites > 0,
    "swept {} models and audited zero guard sites — vacuous",
    models.len()
  );

  assert!(
    failures.is_empty(),
    "fp16 guard sweep failed over {} models / {audited_sites} guard sites:\n\n{}\n",
    models.len(),
    failures.join("\n\n")
  );
}
