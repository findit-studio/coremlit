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
  fs,
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

/// Reads a MIL program into constants and producers.
fn parse_mil(text: &str) -> Graph {
  let mut consts = BTreeMap::new();
  let mut producers = BTreeMap::new();

  for line in text.lines() {
    let line = line.trim();
    let Some(rest) = line.strip_prefix("tensor<") else {
      continue;
    };
    // `fp16, [1, 2999, 29]> var = op(args)[attrs];` — shapes never nest
    // angle brackets, so the first `>` closes the tensor type.
    let (ty, rest) = match rest.split_once('>') {
      Some(parts) => parts,
      None => continue,
    };
    let dtype = ty.split(',').next().unwrap_or("").trim().to_string();

    let Some((var, rest)) = rest.split_once('=') else {
      continue;
    };
    let var = var.trim();
    if var.is_empty() || !var.chars().all(|c| c.is_alphanumeric() || c == '_') {
      continue;
    }

    let rest = rest.trim();
    let Some(open) = rest.find('(') else { continue };
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
    let Some(close) = close else { continue };
    let args = rest[open + 1..close].to_string();
    let attrs = rest[close + 1..].trim().to_string();

    if op == "const"
      && let Some(value) = const_scalar(&attrs)
    {
      consts.insert(var.to_string(), value);
    }
    producers.insert(var.to_string(), Stmt { dtype, op, args });
  }

  Graph { consts, producers }
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
      // `x + eps` — the classic explicit guard.
      "add" => ["y", "x"]
        .iter()
        .find_map(|k| self.value(arg(&stmt.args, k)))
        .map(|c| (c, format!("add(+{c:e})"))),
      "clip" => self
        .value(arg(&stmt.args, "alpha"))
        .map(|lo| (lo, format!("clip(alpha={lo:e})"))),
      "maximum" => ["y", "x"]
        .iter()
        .find_map(|k| self.value(arg(&stmt.args, k)))
        .map(|c| (c, format!("maximum({c:e})"))),
      // A softmax output can underflow to exactly 0 in fp16 long before
      // the log's epsilon is ever added — the decomposed-log_softmax trap.
      "softmax" => Some((0.0, "softmax->log".to_string())),
      "cast" => self.floor(arg(&stmt.args, "x"), depth + 1),
      _ => None,
    }
  }

  /// Every guard site in the graph, in a stable order.
  fn audit(&self) -> Vec<Finding> {
    let mut found = Vec::new();
    for (var, stmt) in &self.producers {
      let eps_kwarg = self.value(arg(&stmt.args, "epsilon"));
      let site = match stmt.op.as_str() {
        // `log` and `rsqrt` always carry an `epsilon` in CoreML MIL, so
        // they are always guard sites — including when that epsilon has
        // already been folded to a literal `0x0p+0`.
        "log" | "rsqrt" => {
          let (floor, guard) = self
            .floor(arg(&stmt.args, "x"), 0)
            .unwrap_or((0.0, "-".into()));
          Some((eps_kwarg.unwrap_or(0.0), floor, guard))
        }
        // A normalization's epsilon is its whole guard.
        "instance_norm" | "layer_norm" | "batch_norm" => eps_kwarg.map(|e| (e, 0.0, "norm".into())),
        // `sqrt` has no epsilon: it is a guard site only when something
        // constant floors its input.
        "sqrt" => self
          .floor(arg(&stmt.args, "x"), 0)
          .map(|(f, g)| (0.0, f, g)),
        // A divide is a guard site when its DIVISOR is const-floored —
        // the `x / (n + eps)` pooling shape.
        "real_div" => self
          .floor(arg(&stmt.args, "y"), 0)
          .map(|(f, g)| (0.0, f, format!("denom:{g}"))),
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
    found
  }
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
    sites: &["real_div/fp32 guard=denom:add(+9.99999993922529e-9) eff=9.99999993922529e-9"],
    note: "Attentive-stat pooling divides by `count + 1e-8`. 1e-8 is 0.168x fp16's smallest \
           subnormal, so on the ANE the divisor guard is zero. Same fp32 artifact, same input, \
           only CpuOnly -> All: cosine collapses to 0.035.",
  },
  KnownDefect {
    path: "speakerkit/wespeaker_v2.mlmodelc",
    sites: &["real_div/fp32 guard=denom:add(+9.99999993922529e-9) eff=9.99999993922529e-9"],
    note: "Same pooling epsilon as wespeaker.mlmodelc.",
  },
  KnownDefect {
    path: "speakerkit/wespeaker_int8.mlmodelc",
    sites: &["real_div/fp32 guard=denom:add(+9.99999993922529e-9) eff=9.99999993922529e-9"],
    note: "Same pooling epsilon as wespeaker.mlmodelc.",
  },
  KnownDefect {
    path: "speakerkit/PLDA.mlmodelc",
    sites: &["sqrt/fp32 guard=clip(alpha=9.999999960041972e-13) eff=9.999999960041972e-13"],
    note: "Normalization clips to 1e-12 before `sqrt`, then divides by it. 1e-12 is 1.7e-5x \
           fp16's smallest subnormal: on the ANE the clip floor is zero, giving sqrt(0) and a \
           divide by zero. Not yet observed in a shipping path (found by this sweep, not by a \
           failure).",
  },
  KnownDefect {
    path: "speakerkit/PldaRho.mlmodelc",
    sites: &["sqrt/fp32 guard=clip(alpha=9.999999960041972e-13) eff=9.999999960041972e-13"],
    note: "Same 1e-12 clip floor as PLDA.mlmodelc.",
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

/// Only `log`/`sqrt`/`rsqrt`/`real_div`/norm sites, not every op.
fn vanishing(mil: &str) -> Vec<String> {
  parse_mil(mil)
    .audit()
    .iter()
    .filter(|f| !f.survives_fp16())
    .map(Finding::render)
    .collect()
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
  let log = audit.iter().find(|f| f.op == "log").expect("a log site");
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
  let log = audit.iter().find(|f| f.op == "log").expect("a log site");
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
fn discover(root: &Path, out: &mut Vec<PathBuf>) {
  let Ok(entries) = fs::read_dir(root) else {
    return;
  };
  for entry in entries.flatten() {
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
      discover(&path, out);
    }
  }
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
  discover(&root, &mut models);
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

    let findings = parse_mil(&text).audit();
    assert!(
      !findings.is_empty(),
      "{rel}: parsed zero guard sites from a {} byte graph — the parser has rotted",
      text.len()
    );
    audited_sites += findings.len();

    let mut vanishing: Vec<String> = findings
      .iter()
      .filter(|f| !f.survives_fp16())
      .map(Finding::render)
      .collect();
    vanishing.sort();
    vanishing.dedup();

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
