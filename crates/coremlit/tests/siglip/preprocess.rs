//! NaFlex preprocessing parity gates.
//!
//! # Status: Wave B (hermetic) + Wave C (model-gated)
//!
//! The pure preprocessing MATH (budget solver against the probe's `spatial_shapes`
//! oracle, antialiased resize, patchify/mask, pos-emb lift, determinism) is
//! proven HERMETICALLY in the in-lib unit tests
//! (`src/embeddings/siglip/image/preprocess/tests.rs`, in the default
//! `cargo test --features siglip`) against hand-derived values. This integration
//! file holds the committed-oracle-fixture arms:
//!
//! - **Wave B** — the committed small oracles (`fixtures/goldens/preprocess.json`,
//!   torch/pillow-generated) cross-checked against an INDEPENDENT in-test
//!   recomputation of the same NaFlex reference math: the budget solver, the u8
//!   PIL resize grids (vs the hand-verified pillow truth), the normalize
//!   constants, the position-lift kernel (within a measured-then-pinned tol), and
//!   the patchify flatten order. Hermetic — no model.
//! - **Wave C** — full-tensor parity: the Rust `pixel_values`/`attention_mask`/
//!   `position_embeddings` vs the staged per-image `.npy` fixtures (via `npyz`),
//!   exact mask equality, bitwise-zero pad rows.

mod common;

use coremlit::embeddings::siglip::{ImageEmbedder, Rgb8Image};
use serde::Deserialize;

const PATCH_SIZE: usize = 16;
const PATCH_BUDGET: usize = 512;
const PATCH_DIM: usize = 3 * 16 * 16;
const EMBED_DIM: usize = 768;

#[derive(Deserialize)]
struct Preprocess {
  budget_table: Vec<BudgetRow>,
  resize_u8: Vec<ResizeCase>,
  normalize: Normalize,
  pos_lift: PosLift,
  patchify: Patchify,
}

#[derive(Deserialize)]
struct BudgetRow {
  height: usize,
  width: usize,
  grid: [usize; 2],
}

#[derive(Deserialize)]
struct ResizeCase {
  src_h: usize,
  src_w: usize,
  dst_h: usize,
  dst_w: usize,
  src: Vec<u8>,
  dst: Vec<u8>,
}

#[derive(Deserialize)]
struct Normalize {
  rescale_factor: f64,
  mean: f32,
  std: f32,
  samples: Vec<NormSample>,
}

#[derive(Deserialize)]
struct NormSample {
  u8: u8,
  f32: f32,
}

#[derive(Deserialize)]
struct PosLift {
  tolerance: f64,
  cases: Vec<PosCase>,
}

#[derive(Deserialize)]
struct PosCase {
  src_h: usize,
  src_w: usize,
  dst_h: usize,
  dst_w: usize,
  input: Vec<f64>,
  values: Vec<f64>,
}

#[derive(Deserialize)]
struct Patchify {
  grid_h: usize,
  grid_w: usize,
  pixels: Vec<i64>,
  patches: Vec<i64>,
}

/// Independent in-test mirror of the Rust `fit_to_patch_budget` binary search,
/// so the committed grid is cross-checked, not merely echoed.
fn fit_to_patch_budget(h: usize, w: usize, patch: usize, budget: usize) -> (usize, usize) {
  const EPS: f64 = 1e-5;
  let pf = patch as f64;
  let scaled = |scale: f64, size: usize| -> usize {
    let s = ((scale * size as f64) / pf).ceil() * pf;
    s.max(pf) as usize
  };
  let grid = |t: usize| t / patch;
  let (mut lo, mut hi) = (EPS / 10.0, 100.0);
  while (hi - lo) >= EPS {
    let s = (lo + hi) / 2.0;
    if grid(scaled(s, h)) * grid(scaled(s, w)) <= budget {
      lo = s;
    } else {
      hi = s;
    }
  }
  (grid(scaled(lo, h)), grid(scaled(lo, w)))
}

/// Per-axis antialiased-bilinear coefficients — the Rust `precompute_coeffs`
/// (triangle filter, support widens by the downscale ratio, `align_corners=false`
/// sample centers).
fn coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f64>)> {
  let scale = in_size as f64 / out_size as f64;
  let fscale = if scale < 1.0 { 1.0 } else { scale };
  let inv = 1.0 / fscale;
  (0..out_size)
    .map(|o| {
      let center = (o as f64 + 0.5) * scale;
      let mut xmin = (center - fscale + 0.5).floor() as isize;
      if xmin < 0 {
        xmin = 0;
      }
      let mut xmax = (center + fscale + 0.5).floor() as isize;
      if xmax > in_size as isize {
        xmax = in_size as isize;
      }
      if xmax <= xmin {
        xmin = xmin.min(in_size as isize - 1).max(0);
        xmax = xmin + 1;
      }
      let start = xmin as usize;
      let taps = (xmax - xmin) as usize;
      let mut ws = Vec::with_capacity(taps);
      let mut sum = 0.0f64;
      for k in 0..taps {
        let x = (start + k) as f64;
        let t = ((x - center + 0.5) * inv).abs();
        let wv = if t < 1.0 { 1.0 - t } else { 0.0 };
        ws.push(wv);
        sum += wv;
      }
      if sum != 0.0 {
        for wv in &mut ws {
          *wv /= sum;
        }
      }
      (start, ws)
    })
    .collect()
}

/// Single-channel f64 antialiased-bilinear resize — the Rust
/// `resize_bilinear_antialias` (separable horizontal-then-vertical, f64 accumulate).
fn resize2d(src: &[f64], sh: usize, sw: usize, dh: usize, dw: usize) -> Vec<f64> {
  let wc = coeffs(sw, dw);
  let mut tmp = vec![0.0f64; sh * dw];
  for y in 0..sh {
    for (ox, (start, ws)) in wc.iter().enumerate() {
      let mut acc = 0.0f64;
      for (k, &wv) in ws.iter().enumerate() {
        acc += wv * src[y * sw + start + k];
      }
      tmp[y * dw + ox] = acc;
    }
  }
  let hc = coeffs(sh, dh);
  let mut out = vec![0.0f64; dh * dw];
  for (oy, (start, ws)) in hc.iter().enumerate() {
    for x in 0..dw {
      let mut acc = 0.0f64;
      for (k, &wv) in ws.iter().enumerate() {
        acc += wv * tmp[(start + k) * dw + x];
      }
      out[oy * dw + x] = acc;
    }
  }
  out
}

/// Small committed oracles vs an INDEPENDENT recomputation of the same NaFlex
/// reference math — hermetic, no model. Budget/normalize/patchify are recomputed
/// exactly; the u8 PIL resize is checked against the hand-verified pillow grids
/// (the same the in-lib E3 oracles pin); the position lift is checked against the
/// reimplemented f64 kernel within the committed measured-then-pinned tolerance.
#[test]
fn small_oracles_match_committed_torch_reference() {
  let bytes = std::fs::read(common::fixture_path("goldens/preprocess.json")).expect("read oracle");
  let pp: Preprocess = serde_json::from_slice(&bytes).expect("parse preprocess.json");

  // (1) budget table — exact vs the independent solver.
  assert!(pp.budget_table.len() >= 10, "budget table too small");
  for row in &pp.budget_table {
    let (gh, gw) = fit_to_patch_budget(row.height, row.width, PATCH_SIZE, PATCH_BUDGET);
    assert_eq!(
      [gh, gw],
      row.grid,
      "budget grid mismatch for {}x{}",
      row.height,
      row.width
    );
    assert!(gh >= 1 && gw >= 1 && gh * gw <= PATCH_BUDGET);
  }
  // the committed 320x240 -> (19,26) cross-link.
  assert!(
    pp.budget_table
      .iter()
      .any(|r| r.height == 240 && r.width == 320 && r.grid == [19, 26]),
    "the 320x240 -> (19,26) budget cross-link must be present"
  );

  // (2) u8 PIL resize — the two upscales exact vs the hand-verified pillow grids;
  // the downscale a symmetric low-pass (per-pass u8 rounding; not recomputable in
  // pure f64, so checked structurally here and exactly by the in-lib E3 oracles).
  #[rustfmt::skip]
  let checker: [u8; 16] = [0,64,191,255, 64,96,159,191, 191,159,96,64, 255,191,64,0];
  #[rustfmt::skip]
  let discriminant: [u8; 16] = [0,0,1,1, 64,64,65,65, 191,191,192,192, 255,255,255,255];
  let mut saw_checker = false;
  let mut saw_discriminant = false;
  let mut saw_downscale = false;
  for case in &pp.resize_u8 {
    assert_eq!(case.src.len(), case.src_h * case.src_w, "resize src length");
    assert_eq!(case.dst.len(), case.dst_h * case.dst_w, "resize dst length");
    if case.src == [0, 255, 255, 0] && (case.dst_h, case.dst_w) == (4, 4) {
      saw_checker = true;
      assert_eq!(
        case.dst, checker,
        "checker PIL grid diverged from hand-verified pillow"
      );
    } else if case.src == [0, 1, 255, 255] && (case.dst_h, case.dst_w) == (4, 4) {
      saw_discriminant = true;
      assert_eq!(
        case.dst, discriminant,
        "per-pass-rounding PIL grid diverged (cells [1][2]=65,[2][2]=192 are the tell)"
      );
    } else if (case.dst_h, case.dst_w) == (2, 2) {
      saw_downscale = true;
      let d = &case.dst;
      assert_eq!(
        (d[0], d[1]),
        (d[2], d[3]),
        "constant columns stay row-identical"
      );
      assert_eq!(
        u16::from(d[0]) + u16::from(d[1]),
        255,
        "antialias downscale must be symmetric about 127.5"
      );
      assert!(
        d[0] > 0 && d[0] < d[1] && d[1] < 255,
        "must low-pass, not hard-subsample"
      );
    }
  }
  assert!(
    saw_checker && saw_discriminant && saw_downscale,
    "missing a resize_u8 case"
  );

  // (3) normalize — exact ((v/255)-0.5)/0.5.
  assert_eq!(pp.normalize.rescale_factor, 1.0 / 255.0);
  assert_eq!(pp.normalize.mean, 0.5);
  assert_eq!(pp.normalize.std, 0.5);
  assert!(pp.normalize.samples.len() >= 4);
  for s in &pp.normalize.samples {
    let want = (f64::from(s.u8) * (1.0 / 255.0)) as f32;
    let want = (want - 0.5) / 0.5;
    assert_eq!(s.f32, want, "normalize sample u8={} diverged", s.u8);
  }

  // (4) position lift — reimplemented f64 antialias-bilinear within the committed
  // measured-then-pinned tolerance vs the torch F.interpolate reference values.
  assert!(pp.pos_lift.tolerance > 0.0 && pp.pos_lift.tolerance < 1e-3);
  assert!(
    pp.pos_lift.cases.len() >= 2,
    "need up- and down-scale pos-lift cases"
  );
  for case in &pp.pos_lift.cases {
    assert_eq!(
      case.input.len(),
      case.src_h * case.src_w,
      "pos input length"
    );
    assert_eq!(
      case.values.len(),
      case.dst_h * case.dst_w,
      "pos values length"
    );
    let got = resize2d(&case.input, case.src_h, case.src_w, case.dst_h, case.dst_w);
    let worst = got
      .iter()
      .zip(&case.values)
      .map(|(a, b)| (a - b).abs())
      .fold(0.0f64, f64::max);
    assert!(
      worst <= pp.pos_lift.tolerance,
      "pos-lift {}x{}->{}x{} worst delta {worst:e} > tol {:e}",
      case.src_h,
      case.src_w,
      case.dst_h,
      case.dst_w,
      pp.pos_lift.tolerance
    );
  }

  // (5) patchify — reimplement the (patch_row, patch_col, py, px, channel) flatten.
  let pt = &pp.patchify;
  let (gh, gw) = (pt.grid_h, pt.grid_w);
  let iw = gw * PATCH_SIZE;
  assert_eq!(pt.pixels.len(), gh * PATCH_SIZE * gw * PATCH_SIZE * 3);
  assert_eq!(pt.patches.len(), gh * gw * PATCH_DIM);
  for ph in 0..gh {
    for pw in 0..gw {
      let row = ph * gw + pw;
      let mut k = 0;
      for py in 0..PATCH_SIZE {
        for px in 0..PATCH_SIZE {
          for c in 0..3 {
            let iy = ph * PATCH_SIZE + py;
            let ix = pw * PATCH_SIZE + px;
            let want = pt.pixels[(iy * iw + ix) * 3 + c];
            assert_eq!(
              pt.patches[row * PATCH_DIM + k],
              want,
              "patchify flatten mismatch at row {row} k {k}"
            );
            k += 1;
          }
        }
      }
    }
  }
}

/// Load a committed/staged `.npy` as a flat `Vec<T>`.
fn load_npy<T: npyz::Deserialize>(path: &std::path::Path) -> Vec<T> {
  let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
  npyz::NpyFile::new(&bytes[..])
    .unwrap_or_else(|e| panic!("parse npy {path:?}: {e}"))
    .into_vec::<T>()
    .unwrap_or_else(|e| panic!("decode npy {path:?}: {e}"))
}

/// Full-tensor parity of the Rust NaFlex tensors against the staged per-image
/// `.npy` fixtures (the slow processor + the official lift). `pixel_values` and
/// the mask must be BITWISE equal (the pillow-fixed-point contract, delegated to
/// colconv). For `position_embeddings`, the lifted REAL rows match within the
/// pinned tolerance while the Rust pad rows are asserted bitwise ZERO — the
/// reference fills pad rows with `resized[0]` (masked, output-invariant), so the
/// pad rows are canonicalized, not compared to the raw dump.
#[test]
#[ignore = "requires staged siglip models (SIGLIP_TEST_MODELS)"]
fn full_tensor_parity_against_staged_npy() {
  const POS_TOL: f32 = 1e-4; // measured-then-pinned; the real-grid lift delta is ~1e-5 class
  let fdir = common::models_dir().join("fixtures").join("preprocess");
  let (images, _texts) = common::golden_corpus();
  // `preprocess` needs the loaded model only for the resolved budget.
  let embedder = ImageEmbedder::from_files(common::vision_model_path(), common::pos_embed_path())
    .expect("load vision");
  let p = embedder.max_num_patches();

  for g in &images {
    let (rgb, w, h) =
      common::decode_png_rgb8(&common::fixture_path(&format!("goldens/{}", g.file)));
    let pre = embedder
      .preprocess(Rgb8Image::new(&rgb, w, h).expect("rgb"))
      .expect("preprocess");

    let pv_ref: Vec<f32> = load_npy(&fdir.join(format!("{}.pixel_values.npy", g.id)));
    let mask_ref: Vec<f32> = load_npy(&fdir.join(format!("{}.attention_mask.npy", g.id)));
    let pos_ref: Vec<f32> = load_npy(&fdir.join(format!("{}.position_embeddings.npy", g.id)));
    let ss: Vec<i64> = load_npy(&fdir.join(format!("{}.spatial_shapes.npy", g.id)));
    assert_eq!(pv_ref.len(), p * PATCH_DIM, "npy pixel_values length");
    assert_eq!(mask_ref.len(), p, "npy mask length");

    // pixel_values + mask: bitwise (whole array, pads included — both zero-pad).
    assert_eq!(
      pre.pixel_values(),
      pv_ref.as_slice(),
      "{}: pixel_values not bitwise-equal",
      g.id
    );
    assert_eq!(
      pre.attention_mask(),
      mask_ref.as_slice(),
      "{}: mask not bitwise-equal",
      g.id
    );

    // spatial_shapes cross-check against the committed golden.
    assert_eq!(
      [ss[0] as usize, ss[1] as usize],
      g.spatial_shapes,
      "{}: spatial_shapes",
      g.id
    );
    let n_real = g.spatial_shapes[0] * g.spatial_shapes[1];

    // position_embeddings: real rows within tol; Rust pad rows bitwise zero.
    let pos = pre.position_embeddings();
    let worst = pos[..n_real * EMBED_DIM]
      .iter()
      .zip(&pos_ref[..n_real * EMBED_DIM])
      .map(|(a, b)| (a - b).abs())
      .fold(0.0f32, f32::max);
    assert!(
      worst <= POS_TOL,
      "{}: pos-emb real-row worst delta {worst:e} > {POS_TOL:e}",
      g.id
    );
    assert!(
      pos[n_real * EMBED_DIM..].iter().all(|&v| v == 0.0),
      "{}: Rust position_embeddings pad rows must be bitwise zero",
      g.id
    );
  }
}
