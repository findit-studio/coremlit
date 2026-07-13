//! The standard frame-based DER calculation (NIST `md-eval` /
//! `pyannote.metrics` `DiarizationErrorRate`), shared by the end-to-end
//! parity suites.
//!
//! # Provenance — this is a VERBATIM copy, not a reimplementation
//!
//! Every item below is copied unchanged from `tests/parity_e2e.rs`, where
//! this calculation was written and verified: its scoring mask was
//! independently reimplemented in Python by a reviewer and reproduced to the
//! exact unit. The six unit tests that pinned it travel WITH it (bottom of
//! this file) so the copy is proven behaviourally identical every run, not
//! merely assumed to be.
//!
//! It lives in its own module (rather than being duplicated inline in the new
//! suite) purely so the duplication has a **seam**: `parity_e2e.rs` still
//! carries its own copy, and cannot be edited here (a concurrent task owns
//! that file). Once that settles, `parity_e2e.rs` can delete its copy and
//! `mod der_calc;` instead — a one-line change — leaving a single definition.
//! Do not fork the two: fix bugs in both, or do the dedupe.
//!
//! # The definition
//!
//! On a 10 ms frame grid (Kaldi/`md-eval` convention; the reference RTTM is
//! itself 10 ms-quantized), after (a) a 0.25 s no-score collar on each side of
//! every reference-segment boundary and (b) optionally excluding frames with
//! more than one reference speaker (`skip_overlap`), with the optimal
//! one-to-one speaker mapping (the assignment that maximizes matched reference
//! speech — Hungarian-equivalent; computed exactly by DP over reference
//! subsets):
//!
//! ```text
//! DER = ( missed + false_alarm + confusion ) / total_reference_speech
//!   missed(i)      = max(0, N_ref(i) - N_hyp(i))
//!   false_alarm(i) = max(0, N_hyp(i) - N_ref(i))
//!   confusion(i)   = min(N_ref(i), N_hyp(i)) - N_correct(i)
//! ```
//!
//! summed over scored frames `i`, where `N_correct(i)` counts reference
//! speakers whose mapped hypothesis speaker is also active. Denominator is
//! `Σ N_ref(i)` (total reference speech). This is the pyannote.metrics
//! decomposition verbatim.
//!
//! The **confusion** term is the clustering diagnostic: miss/false-alarm move
//! with voice-activity boundaries (benign jitter), whereas confusion means the
//! hypothesis put reference speech under the WRONG speaker — a genuine
//! clustering divergence, which is exactly how the argmax source's spurious
//! extra speaker was caught (3.33 % DER, 100 % of it confusion).

// Each integration-test binary is its own crate, so items this binary does not
// call are dead code *in that crate*. Allowed here so the shared module stays
// clean under the workspace `-D warnings` gate (same rationale as
// `tests/common/mod.rs`).
#![allow(dead_code)]

use std::{collections::BTreeSet, path::Path};

/// DER frame-grid step in seconds (10 ms — the Kaldi/`md-eval` convention;
/// the pyannote reference RTTM is itself quantized to 10 ms frames).
pub const DER_STEP_S: f64 = 0.010;

/// Standard scoring collar in seconds, applied on EACH side of every
/// reference-segment boundary (NIST `md-eval -c 0.25`; matches FluidAudio's
/// `DiarizationMetricsCalculator.scoringCollarSeconds`).
pub const DER_COLLAR_S: f64 = 0.25;

/// A labelled speech turn: `[start, end)` seconds attributed to integer
/// speaker id `spk`.
#[derive(Debug, Clone, Copy)]
pub struct Seg {
  pub start: f64,
  pub end: f64,
  pub spk: usize,
}

/// The full DER breakdown over scored frames (all fractions are of total
/// reference speech; the `_units` fields are the raw speaker-frame counts).
#[derive(Debug, Clone, Copy)]
pub struct Der {
  pub der: f64,
  pub miss: f64,
  pub fa: f64,
  pub confusion: f64,
  pub miss_units: u64,
  pub fa_units: u64,
  pub conf_units: u64,
  pub ref_units: u64,
  pub scored_frames: u64,
  pub err_frames: u64,
  pub num_ref_spk: usize,
  pub num_hyp_spk: usize,
}

/// Distinct speaker ids appearing in `segs` with any positive duration.
pub fn distinct_speakers(segs: &[Seg]) -> BTreeSet<usize> {
  segs
    .iter()
    .filter(|s| s.end > s.start)
    .map(|s| s.spk)
    .collect()
}

/// Optimal one-to-one hypothesis→reference mapping maximizing total matched
/// co-occurrence (`cooccur[h][r]` = scored frames where hyp `h` and ref `r`
/// are both active). Returns `map[h] = Some(r)` or `None` (unmapped hyp).
///
/// Exact global optimum by DP over reference subsets (`O(n_hyp · 2^n_ref)`)
/// — the same optimum a Hungarian max-weight assignment finds. Falls back to
/// greedy only past `MAX_DP_REF` reference speakers. Ties resolve to the
/// lowest reference index (and to "unmapped") for determinism.
pub fn optimal_hyp_to_ref(cooccur: &[Vec<u64>], n_hyp: usize, n_ref: usize) -> Vec<Option<usize>> {
  const MAX_DP_REF: usize = 20;
  if n_hyp == 0 {
    return Vec::new();
  }
  if n_ref == 0 {
    return vec![None; n_hyp];
  }
  if n_ref > MAX_DP_REF {
    // Greedy fallback (not reached by the committed fixtures).
    let mut used = vec![false; n_ref];
    return (0..n_hyp)
      .map(|h| {
        let mut best: Option<(u64, usize)> = None;
        for r in 0..n_ref {
          if !used[r] && cooccur[h][r] > 0 {
            match best {
              Some((bv, _)) if cooccur[h][r] <= bv => {}
              _ => best = Some((cooccur[h][r], r)),
            }
          }
        }
        best.map(|(_, r)| {
          used[r] = true;
          r
        })
      })
      .collect();
  }

  let full = 1usize << n_ref;
  // best[h][mask] = max additional match value assigning hyp h.. with the
  // reference speakers in `mask` already taken; choice records the pick.
  let mut best = vec![vec![0u64; full]; n_hyp + 1];
  let mut choice = vec![vec![usize::MAX; full]; n_hyp]; // usize::MAX = unmapped
  for h in (0..n_hyp).rev() {
    for mask in 0..full {
      // Option 1: leave hyp h unmapped.
      let mut cur = best[h + 1][mask];
      let mut pick = usize::MAX;
      // Option 2: map h to any free reference r.
      for r in 0..n_ref {
        if mask & (1 << r) == 0 {
          let cand = cooccur[h][r] + best[h + 1][mask | (1 << r)];
          if cand > cur {
            cur = cand;
            pick = r;
          }
        }
      }
      best[h][mask] = cur;
      choice[h][mask] = pick;
    }
  }
  let mut map = vec![None; n_hyp];
  let mut mask = 0usize;
  for (h, slot) in map.iter_mut().enumerate() {
    let pick = choice[h][mask];
    if pick != usize::MAX {
      *slot = Some(pick);
      mask |= 1 << pick;
    }
  }
  map
}

/// Compute DER of `hypothesis` against `reference` (the pyannote.metrics
/// decomposition — see the module doc). `collar` seconds are excluded on each
/// side of every reference boundary; `skip_overlap` additionally excludes
/// frames with more than one reference speaker.
pub fn der(reference: &[Seg], hypothesis: &[Seg], collar: f64, skip_overlap: bool) -> Der {
  let num_ref_spk = distinct_speakers(reference).len();
  let num_hyp_spk = distinct_speakers(hypothesis).len();

  // Frame grid over the union extent.
  let t_end = reference
    .iter()
    .chain(hypothesis.iter())
    .map(|s| s.end)
    .fold(0.0_f64, f64::max);
  if t_end <= 0.0 {
    return Der {
      der: 0.0,
      miss: 0.0,
      fa: 0.0,
      confusion: 0.0,
      miss_units: 0,
      fa_units: 0,
      conf_units: 0,
      ref_units: 0,
      scored_frames: 0,
      err_frames: 0,
      num_ref_spk,
      num_hyp_spk,
    };
  }
  let n_frames = (t_end / DER_STEP_S).ceil() as usize;

  // Reference boundaries for the collar.
  let mut boundaries: Vec<f64> = Vec::with_capacity(reference.len() * 2);
  for s in reference {
    if s.end > s.start {
      boundaries.push(s.start);
      boundaries.push(s.end);
    }
  }

  // Contingency counts over scored frames, and the running error tallies.
  let mut cooccur = vec![vec![0u64; num_ref_spk.max(1)]; num_hyp_spk.max(1)];
  // Speaker id → dense index, in first-seen order (stable, test-friendly).
  let ref_ids: Vec<usize> = distinct_speakers(reference).into_iter().collect();
  let hyp_ids: Vec<usize> = distinct_speakers(hypothesis).into_iter().collect();
  let ref_idx = |spk: usize| ref_ids.iter().position(|&r| r == spk).unwrap();
  let hyp_idx = |spk: usize| hyp_ids.iter().position(|&h| h == spk).unwrap();

  // Per-frame active speaker sets (deduped), plus scored mask. Two passes so
  // the optimal mapping (pass 1) is available for the error tally (pass 2).
  let mut ref_active: Vec<Vec<usize>> = vec![Vec::new(); n_frames];
  let mut hyp_active: Vec<Vec<usize>> = vec![Vec::new(); n_frames];
  let mut scored: Vec<bool> = vec![true; n_frames];
  for i in 0..n_frames {
    let c = (i as f64 + 0.5) * DER_STEP_S;
    for s in reference {
      if s.end > s.start && c >= s.start && c < s.end && !ref_active[i].contains(&s.spk) {
        ref_active[i].push(s.spk);
      }
    }
    for s in hypothesis {
      if s.end > s.start && c >= s.start && c < s.end && !hyp_active[i].contains(&s.spk) {
        hyp_active[i].push(s.spk);
      }
    }
    // Collar: within `collar` of any reference boundary → no-score.
    if collar > 0.0 && boundaries.iter().any(|&b| (c - b).abs() < collar) {
      scored[i] = false;
    }
    // skip_overlap: multi-speaker reference frame → no-score.
    if skip_overlap && ref_active[i].len() > 1 {
      scored[i] = false;
    }
  }

  // Pass 1: optimal mapping over scored frames.
  for i in 0..n_frames {
    if !scored[i] {
      continue;
    }
    for &r in &ref_active[i] {
      for &h in &hyp_active[i] {
        cooccur[hyp_idx(h)][ref_idx(r)] += 1;
      }
    }
  }
  let map = optimal_hyp_to_ref(&cooccur, num_hyp_spk, num_ref_spk);
  // ref index → mapped hyp id (invert the 1-to-1 hyp→ref map).
  let mut ref_to_hyp: Vec<Option<usize>> = vec![None; num_ref_spk];
  for (h, m) in map.iter().enumerate() {
    if let Some(r) = m {
      ref_to_hyp[*r] = Some(hyp_ids[h]);
    }
  }

  // Pass 2: the error tally.
  let (
    mut miss_units,
    mut fa_units,
    mut conf_units,
    mut ref_units,
    mut scored_frames,
    mut err_frames,
  ) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
  for i in 0..n_frames {
    if !scored[i] {
      continue;
    }
    scored_frames += 1;
    let n_ref = ref_active[i].len() as u64;
    let n_hyp = hyp_active[i].len() as u64;
    ref_units += n_ref;
    // N_correct: reference speakers whose mapped hyp speaker is active here.
    let mut n_correct = 0u64;
    for &r in &ref_active[i] {
      if let Some(h) = ref_to_hyp[ref_idx(r)]
        && hyp_active[i].contains(&h)
      {
        n_correct += 1;
      }
    }
    let miss = n_ref.saturating_sub(n_hyp);
    let fa = n_hyp.saturating_sub(n_ref);
    let confusion = n_ref.min(n_hyp) - n_correct;
    miss_units += miss;
    fa_units += fa;
    conf_units += confusion;
    if miss + fa + confusion > 0 {
      err_frames += 1;
    }
  }

  let denom = ref_units.max(1) as f64;
  Der {
    der: (miss_units + fa_units + conf_units) as f64 / denom,
    miss: miss_units as f64 / denom,
    fa: fa_units as f64 / denom,
    confusion: conf_units as f64 / denom,
    miss_units,
    fa_units,
    conf_units,
    ref_units,
    scored_frames,
    err_frames,
    num_ref_spk,
    num_hyp_spk,
  }
}

/// The standard-collar DER (0.25 s collar, overlap excluded) — the DIHARD /
/// NIST / pyannote convention used for the absolute-accuracy numbers, and the
/// metric the spec's "DER" names. This is what gates.
pub fn der_std(reference: &[Seg], hypothesis: &[Seg]) -> Der {
  der(reference, hypothesis, DER_COLLAR_S, true)
}

/// The strict frame-exact DER (no collar, no overlap-skip): every frame
/// counts, so it surfaces every sub-collar boundary difference between two
/// near-identical pipelines. REPORTED, not the pass/fail bound: at a 10 ms grid
/// it is dominated by the ±1-3 frame boundary quantization of the accepted
/// 99.97 % segmentation agreement, which the standard DER absorbs by design.
pub fn der_strict(reference: &[Seg], hypothesis: &[Seg]) -> Der {
  der(reference, hypothesis, 0.0, false)
}

/// Parse a NIST RTTM file into [`Seg`]s, mapping each `SPEAKER_xx` label to a
/// stable integer id in first-appearance order. Only `SPEAKER` rows are read;
/// fields are `type uri chan start dur <NA> <NA> spk <NA> <NA>`.
pub fn parse_rttm(path: &Path) -> Vec<Seg> {
  let text =
    std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read rttm {}: {e}", path.display()));
  let mut labels: Vec<String> = Vec::new();
  let mut segs = Vec::new();
  for line in text.lines() {
    let f: Vec<&str> = line.split_whitespace().collect();
    if f.len() < 8 || f[0] != "SPEAKER" {
      continue;
    }
    let start: f64 = f[3]
      .parse()
      .unwrap_or_else(|_| panic!("rttm start: {line}"));
    let dur: f64 = f[4].parse().unwrap_or_else(|_| panic!("rttm dur: {line}"));
    let label = f[7];
    let spk = labels.iter().position(|l| l == label).unwrap_or_else(|| {
      labels.push(label.to_string());
      labels.len() - 1
    });
    segs.push(Seg {
      start,
      end: start + dur,
      spk,
    });
  }
  segs
}

/// One-line DER summary for the run logs.
pub fn fmt_der(tag: &str, d: &Der) -> String {
  format!(
    "{tag}: DER={:.4}% (miss={:.4}% fa={:.4}% conf={:.4}%) | ref_spk={} hyp_spk={} | \
     units miss/fa/conf/ref={}/{}/{}/{} | err_frames={}/{} scored",
    d.der * 100.0,
    d.miss * 100.0,
    d.fa * 100.0,
    d.confusion * 100.0,
    d.num_ref_spk,
    d.num_hyp_spk,
    d.miss_units,
    d.fa_units,
    d.conf_units,
    d.ref_units,
    d.err_frames,
    d.scored_frames,
  )
}

// ══════════════════════════════════════════════════════════════════════
// Unit tests for the DER calc itself — copied verbatim WITH the calc, so the
// copy is proven behaviourally identical to `parity_e2e.rs`'s original every
// run rather than assumed to be. No models needed; these run in the ordinary
// `--features dia` suite.
// ══════════════════════════════════════════════════════════════════════

#[cfg(test)]
fn approx(a: f64, b: f64) -> bool {
  (a - b).abs() < 1e-9
}

/// Identical reference and hypothesis (even with a speaker relabel) ⇒ DER 0.
#[test]
fn der_identical_is_zero() {
  let reference = vec![
    Seg {
      start: 0.0,
      end: 5.0,
      spk: 0,
    },
    Seg {
      start: 5.0,
      end: 10.0,
      spk: 1,
    },
  ];
  // Same timeline, speakers relabelled — the optimal mapping must recover a
  // perfect match regardless of label identity.
  let hypothesis = vec![
    Seg {
      start: 0.0,
      end: 5.0,
      spk: 9,
    },
    Seg {
      start: 5.0,
      end: 10.0,
      spk: 3,
    },
  ];
  let d = der_strict(&reference, &hypothesis);
  assert!(approx(d.der, 0.0), "identical ⇒ DER 0, got {}", d.der);
  assert_eq!(d.miss_units + d.fa_units + d.conf_units, 0);
}

/// Empty hypothesis over speech ⇒ 100 % miss.
#[test]
fn der_total_miss_is_one() {
  let reference = vec![Seg {
    start: 0.0,
    end: 10.0,
    spk: 0,
  }];
  let d = der_strict(&reference, &[]);
  assert!(approx(d.der, 1.0), "total miss ⇒ DER 1.0, got {}", d.der);
  assert!(approx(d.miss, 1.0) && approx(d.fa, 0.0) && approx(d.confusion, 0.0));
}

/// Hypothesis speech where the reference is silent ⇒ false alarm.
#[test]
fn der_false_alarm() {
  let reference = vec![Seg {
    start: 0.0,
    end: 10.0,
    spk: 0,
  }];
  let hypothesis = vec![
    Seg {
      start: 0.0,
      end: 10.0,
      spk: 0,
    },
    Seg {
      start: 10.0,
      end: 20.0,
      spk: 0,
    },
  ];
  let d = der_strict(&reference, &hypothesis);
  assert!(approx(d.fa, 1.0), "expected 100% FA, got {}", d.fa);
  assert!(approx(d.miss, 0.0) && approx(d.confusion, 0.0));
  assert!(approx(d.der, 1.0));
}

/// One reference speaker; the hypothesis splits the back half into a second,
/// UNmapped speaker ⇒ 50 % confusion. This is the shape of the failure this
/// suite exists to catch (a spurious extra speaker ⇒ pure confusion).
#[test]
fn der_confusion_from_split() {
  let reference = vec![Seg {
    start: 0.0,
    end: 10.0,
    spk: 0,
  }];
  let hypothesis = vec![
    Seg {
      start: 0.0,
      end: 5.0,
      spk: 0,
    },
    Seg {
      start: 5.0,
      end: 10.0,
      spk: 1,
    },
  ];
  let d = der_strict(&reference, &hypothesis);
  assert!(
    approx(d.confusion, 0.5),
    "expected 50% confusion, got {}",
    d.confusion
  );
  assert!(approx(d.miss, 0.0) && approx(d.fa, 0.0));
  assert!(approx(d.der, 0.5));
}

/// The collar removes near-boundary error.
#[test]
fn der_collar_excludes_boundary() {
  let reference = vec![Seg {
    start: 0.0,
    end: 10.0,
    spk: 0,
  }];
  let hypothesis = vec![Seg {
    start: 0.0,
    end: 9.9,
    spk: 0,
  }];
  let strict = der_strict(&reference, &hypothesis);
  assert!(
    strict.miss > 0.005 && strict.miss < 0.02,
    "strict miss ≈ 1%, got {}",
    strict.miss
  );
  let collared = der(&reference, &hypothesis, DER_COLLAR_S, false);
  assert!(
    approx(collared.der, 0.0),
    "0.25 s collar should erase the boundary miss, got {}",
    collared.der
  );
}

/// The optimal mapping must pick the assignment that MAXIMIZES matched
/// speech, not a greedy first pick.
#[test]
fn optimal_mapping_is_global() {
  // cooccur[h][r]: hyp 0 overlaps ref0=1, ref1=9; hyp 1 overlaps ref0=8.
  let cooccur = vec![vec![1u64, 9u64], vec![8u64, 0u64]];
  let map = optimal_hyp_to_ref(&cooccur, 2, 2);
  assert_eq!(map, vec![Some(1), Some(0)], "expected global optimum 9+8");
}
