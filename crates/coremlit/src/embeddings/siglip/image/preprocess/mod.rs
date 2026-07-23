//! NaFlex vision preprocessing — pure host-side math, no CoreML.
//!
//! Ports the SigLIP2 NaFlex image pipeline (transformers
//! `image_processing_siglip2.py` + the `resize_positional_embeddings` lift from
//! `modeling_siglip2.py`) into dependency-free Rust, in the stages the vision
//! graph's `[1, P, 768]` `pixel_values` / `position_embeddings` / `[1, P]`
//! `attention_mask` contract needs:
//!
//! 1. [`fit_to_patch_budget`] — the aspect-preserving budget solver: binary-
//!    search the largest multiple-of-`patch` target whose patch count `h_p·w_p`
//!    fits `P`.
//! 2. [`resize_bilinear_antialias`] — the shared antialiased-bilinear resample
//!    (PIL/torch `F.interpolate(mode="bilinear", antialias=True,
//!    align_corners=False)` coefficients), generic over channel count, used both
//!    to resize the image and to lift the position-embedding grid.
//! 3. [`normalize_pixel`] — rescale + normalize `((x/255) − 0.5)/0.5`.
//! 4. [`patchify`] — reshape the resized+normalized image to `[P, 768]` patch
//!    rows plus the `[P]` real/pad mask.
//! 5. [`lift_position_embeddings`] — resize the base `16×16×768` grid to the
//!    `(h_p, w_p)` grid, flatten row-major, and zero-pad to `[P, 768]`.
//!
//! The pixel-normalization constants and patch flatten order are pinned by
//! committed fixtures (`tests/siglip/fixtures/goldens/preprocess.json`, Wave B)
//! and the full-tensor parity against staged `.npy` fixtures (Wave C); the pure
//! stage math here is proven hermetically in `tests.rs`.
//!
//! Accumulation is in `f64` (weights and dot-products) for determinism; the
//! measured-then-pinned resize tolerance (Wave B3) absorbs any residual delta
//! against the torch reference's working precision.

use crate::embeddings::siglip::{embedding::EMBEDDING_DIM, error::Error};

/// Vision patch side in pixels (a `16×16` patch). Architecture constant of the
/// `patch16` model — not a resolved parameter (unlike the patch budget `P`).
pub(crate) const PATCH_SIZE: usize = 16;

/// RGB channel count of a decoded image.
pub(crate) const CHANNELS: usize = 3;

/// Flattened per-patch dimension `CHANNELS · PATCH_SIZE · PATCH_SIZE = 3·16·16 =
/// 768` — the `pixel_values` inner dimension. Coincides with [`EMBEDDING_DIM`]
/// for `base-patch16`, but is a distinct quantity.
pub(crate) const PATCH_DIM: usize = CHANNELS * PATCH_SIZE * PATCH_SIZE;

/// Side of the base position-embedding grid: the model's `num_patches = 256`
/// position table reshaped to `16×16` (per the probe). Architecture constant.
pub(crate) const POS_GRID_SIDE: usize = 16;

/// Element count of the base position-embedding grid,
/// `POS_GRID_SIDE · POS_GRID_SIDE · EMBEDDING_DIM = 16·16·768 = 196 608` f32.
pub(crate) const POS_EMBED_ELEMS: usize = POS_GRID_SIDE * POS_GRID_SIDE * EMBEDDING_DIM;

/// Byte length of the raw little-endian f32 base position-embedding grid sidecar
/// (`pos_embed_16x16x768.f32le.bin`): `POS_EMBED_ELEMS · 4 = 786 432` bytes.
/// Derived from the dimensional constants so it stays correct by construction
/// (the load-time hard-validation of D5).
pub(crate) const POS_EMBED_BYTES: usize = POS_EMBED_ELEMS * 4;

/// Pixel rescale factor `1/255` (`preprocessor_config.json` `rescale_factor`).
const RESCALE_FACTOR: f32 = 1.0 / 255.0;
/// Per-channel normalization mean `0.5` (`image_mean`).
const IMAGE_MEAN: f32 = 0.5;
/// Per-channel normalization std `0.5` (`image_std`).
const IMAGE_STD: f32 = 0.5;

/// Aspect-preserving patch-budget solver: the largest multiple-of-`patch`
/// target size whose patch grid `h_p · w_p` fits `budget`, returned as the
/// `(grid_height, grid_width)` patch counts (the processor's `spatial_shapes`).
///
/// A direct port of transformers `get_image_size_for_max_num_patches`: binary-
/// search a uniform scale in `[eps/10, 100]` to `eps = 1e-5`, where each
/// candidate maps a dimension to `max(patch, ceil(scale·size/patch)·patch)`, and
/// keep the largest scale whose grid fits the budget. Uniform scaling preserves
/// aspect ratio; the `max(patch, …)` clamp guarantees at least one patch per
/// dimension (so an extreme aspect keeps a `1`-patch short side). The result is
/// (essentially) a function of aspect ratio and `budget`, near-independent of
/// absolute pixel size — a small image is upscaled to fill the budget exactly as
/// NaFlex does.
///
/// `patch` and `budget` must be non-zero (the caller resolves `budget = P` from
/// the loaded model; `patch = PATCH_SIZE`).
pub(crate) fn fit_to_patch_budget(
  image_height: usize,
  image_width: usize,
  patch: usize,
  budget: usize,
) -> (usize, usize) {
  const EPS: f64 = 1e-5;
  let patch_f = patch as f64;

  let scaled = |scale: f64, size: usize| -> usize {
    let s = (scale * size as f64) / patch_f;
    let s = s.ceil() * patch_f;
    let s = s.max(patch_f);
    s as usize
  };
  let grid = |target: usize| target / patch;

  let mut scale_min = EPS / 10.0;
  let mut scale_max = 100.0;
  while (scale_max - scale_min) >= EPS {
    let scale = (scale_min + scale_max) / 2.0;
    let th = scaled(scale, image_height);
    let tw = scaled(scale, image_width);
    if grid(th) * grid(tw) <= budget {
      scale_min = scale;
    } else {
      scale_max = scale;
    }
  }
  let th = scaled(scale_min, image_height);
  let tw = scaled(scale_min, image_width);
  (grid(th), grid(tw))
}

/// The antialiased-bilinear filter weight (triangle): `max(0, 1 − |t|)`.
fn triangle(t: f64) -> f64 {
  let t = t.abs();
  if t < 1.0 { 1.0 - t } else { 0.0 }
}

/// Per-output-sample resample coefficients for one axis (`in_size → out_size`),
/// the PIL/torch `precompute_coeffs` for a bilinear filter with `antialias`:
/// the triangle support widens by the downscale ratio (`filterscale`) so
/// downsampling low-passes; upsampling (`scale ≤ 1`) keeps unit support (plain
/// bilinear). `align_corners=False` sample centers `(o + 0.5)·scale`. Each entry
/// is `(start_input_index, normalized_weights)`.
fn precompute_coeffs(in_size: usize, out_size: usize) -> Vec<(usize, Vec<f64>)> {
  let scale = in_size as f64 / out_size as f64;
  let filterscale = if scale < 1.0 { 1.0 } else { scale };
  let support = filterscale; // triangle filter support is 1.0
  let inv_filterscale = 1.0 / filterscale;

  let mut coeffs = Vec::with_capacity(out_size);
  for o in 0..out_size {
    let center = (o as f64 + 0.5) * scale;

    let mut xmin = (center - support + 0.5).floor() as isize;
    if xmin < 0 {
      xmin = 0;
    }
    let mut xmax = (center + support + 0.5).floor() as isize;
    if xmax > in_size as isize {
      xmax = in_size as isize;
    }
    // Guarantee at least one in-bounds tap (degenerate only at extreme edges).
    if xmax <= xmin {
      xmin = xmin.min(in_size as isize - 1).max(0);
      xmax = xmin + 1;
    }
    let start = xmin as usize;
    let taps = (xmax - xmin) as usize;

    let mut weights = Vec::with_capacity(taps);
    let mut sum = 0.0f64;
    for k in 0..taps {
      let x = (start + k) as f64;
      let w = triangle((x - center + 0.5) * inv_filterscale);
      weights.push(w);
      sum += w;
    }
    if sum != 0.0 {
      for w in &mut weights {
        *w /= sum;
      }
    }
    coeffs.push((start, weights));
  }
  coeffs
}

/// Antialiased-bilinear resample of an `(src_h, src_w, channels)` row-major
/// buffer to `(dst_h, dst_w, channels)`. Separable: resize width, then height
/// (the PIL horizontal-then-vertical order), accumulating in `f64`.
///
/// Shared by the image resize (`channels = 3`) and the position-embedding grid
/// lift (`channels = 768`). Panics only on an inconsistent `src` length
/// (`src.len() != src_h·src_w·channels`) — a caller-internal invariant.
pub(crate) fn resize_bilinear_antialias(
  src: &[f32],
  src_h: usize,
  src_w: usize,
  channels: usize,
  dst_h: usize,
  dst_w: usize,
) -> Vec<f32> {
  assert_eq!(
    src.len(),
    src_h * src_w * channels,
    "resize src length inconsistent with src_h·src_w·channels"
  );

  // Horizontal pass: (src_h, src_w, C) → (src_h, dst_w, C), kept in f64 to avoid
  // an intermediate rounding between the two separable passes.
  let wc = precompute_coeffs(src_w, dst_w);
  let mut tmp = vec![0.0f64; src_h * dst_w * channels];
  for y in 0..src_h {
    for (ox, (start, weights)) in wc.iter().enumerate() {
      for c in 0..channels {
        let mut acc = 0.0f64;
        for (k, &wk) in weights.iter().enumerate() {
          acc += wk * src[(y * src_w + start + k) * channels + c] as f64;
        }
        tmp[(y * dst_w + ox) * channels + c] = acc;
      }
    }
  }

  // Vertical pass: (src_h, dst_w, C) → (dst_h, dst_w, C).
  let hc = precompute_coeffs(src_h, dst_h);
  let mut out = vec![0.0f32; dst_h * dst_w * channels];
  for (oy, (start, weights)) in hc.iter().enumerate() {
    for x in 0..dst_w {
      for c in 0..channels {
        let mut acc = 0.0f64;
        for (k, &wk) in weights.iter().enumerate() {
          acc += wk * tmp[((start + k) * dst_w + x) * channels + c];
        }
        out[(oy * dst_w + x) * channels + c] = acc as f32;
      }
    }
  }
  out
}

/// Rescale + normalize one pixel channel value from `[0, 255]` to
/// `((x/255) − 0.5)/0.5 ∈ [−1, 1]` — the SigLIP2 processor's `do_rescale` +
/// `do_normalize` (mean `0.5`, std `0.5`), evaluated in that order.
pub(crate) fn normalize_pixel(x: f32) -> f32 {
  (x * RESCALE_FACTOR - IMAGE_MEAN) / IMAGE_STD
}

/// Reshape a resized+normalized `(grid_h·PATCH_SIZE, grid_w·PATCH_SIZE, 3)`
/// image (row-major, RGB-interleaved f32) into the `[budget, PATCH_DIM]`
/// `pixel_values` rows plus the `[budget]` real/pad `attention_mask` (1.0 real,
/// 0.0 pad).
///
/// Patch rows are in `(patch_row, patch_col)` row-major order; each row is the
/// patch's pixels flattened `(py, px, channel)` — `py`-major, then `px`, then
/// the RGB channel — matching the transformers reshape. Rows `grid_h·grid_w ..
/// budget` are zero-padded (mask `0.0`).
///
/// # Errors
/// [`Error::PatchCount`] if `grid_h · grid_w > budget` (a solver/plumbing bug —
/// the budget solver caps this by construction).
pub(crate) fn patchify(
  image_hwc: &[f32],
  grid_h: usize,
  grid_w: usize,
  budget: usize,
) -> Result<(Vec<f32>, Vec<f32>), Error> {
  let n_real = grid_h * grid_w;
  if n_real > budget {
    return Err(Error::PatchCount {
      got: n_real,
      max: budget,
    });
  }
  let img_w = grid_w * PATCH_SIZE;
  let mut pixel_values = vec![0.0f32; budget * PATCH_DIM];
  let mut mask = vec![0.0f32; budget];

  for ph in 0..grid_h {
    for pw in 0..grid_w {
      let row = ph * grid_w + pw;
      let dst_base = row * PATCH_DIM;
      let mut k = 0;
      for py in 0..PATCH_SIZE {
        let iy = ph * PATCH_SIZE + py;
        for px in 0..PATCH_SIZE {
          let ix = pw * PATCH_SIZE + px;
          let src_base = (iy * img_w + ix) * CHANNELS;
          for c in 0..CHANNELS {
            pixel_values[dst_base + k] = image_hwc[src_base + c];
            k += 1;
          }
        }
      }
      mask[row] = 1.0;
    }
  }
  Ok((pixel_values, mask))
}

/// Parse the raw little-endian f32 base position-embedding grid sidecar into a
/// `POS_EMBED_ELEMS`-length `Vec<f32>` (`16×16×768`, row-major), hard-validating
/// the byte length (D5).
///
/// # Errors
/// [`Error::PosEmbedLength`] if `bytes.len() != POS_EMBED_BYTES`.
pub(crate) fn parse_base_pos_grid(bytes: &[u8]) -> Result<Vec<f32>, Error> {
  if bytes.len() != POS_EMBED_BYTES {
    return Err(Error::PosEmbedLength {
      got: bytes.len(),
      expected: POS_EMBED_BYTES,
    });
  }
  let (chunks, _rest) = bytes.as_chunks::<4>(); // len is an exact multiple of 4
  let grid = chunks.iter().map(|&c| f32::from_le_bytes(c)).collect();
  Ok(grid)
}

/// Lift the base `16×16×768` position grid to the `(grid_h, grid_w)` patch grid
/// and flatten to the `[budget, 768]` `position_embeddings` input: resize the
/// grid with the shared antialiased-bilinear kernel, take its row-major
/// `(grid_h·grid_w, 768)` flattening (the resize already emits that layout), and
/// zero-pad the remaining `budget − grid_h·grid_w` rows.
///
/// `base_grid` must be `POS_EMBED_ELEMS` long (as [`parse_base_pos_grid`]
/// returns) and `grid_h · grid_w ≤ budget` (the budget solver guarantees it;
/// excess is defensively truncated rather than panicking).
pub(crate) fn lift_position_embeddings(
  base_grid: &[f32],
  grid_h: usize,
  grid_w: usize,
  budget: usize,
) -> Vec<f32> {
  let resized = resize_bilinear_antialias(
    base_grid,
    POS_GRID_SIDE,
    POS_GRID_SIDE,
    EMBEDDING_DIM,
    grid_h,
    grid_w,
  );
  let mut out = vec![0.0f32; budget * EMBEDDING_DIM];
  let copy_len = resized.len().min(out.len());
  out[..copy_len].copy_from_slice(&resized[..copy_len]);
  out
}

/// The three vision-graph input tensors produced from one decoded image, plus
/// the resolved `(grid_h, grid_w)` patch grid.
pub(crate) struct VisionInputs {
  /// Patchified pixels `[budget · PATCH_DIM]` (the `pixel_values` input).
  pub pixel_values: Vec<f32>,
  /// Real/pad patch mask `[budget]` (1.0 real, 0.0 pad; the `attention_mask`
  /// input — f32, per the graph contract).
  pub attention_mask: Vec<f32>,
  /// Lifted position embeddings `[budget · EMBEDDING_DIM]` (the
  /// `position_embeddings` input).
  pub position_embeddings: Vec<f32>,
  /// The resolved patch grid `(h_p, w_p)` (the processor's `spatial_shapes`).
  /// Consumed host-side (mask + pos-emb are already built from it) rather than
  /// fed to the graph, so production `embed` does not read it back; the hermetic
  /// pipeline tests assert it.
  #[allow(dead_code)]
  pub grid: (usize, usize),
}

/// The full model-free NaFlex image pipeline: decoded RGB8 in, the three vision-
/// graph input tensors out. Fits the image to the patch `budget`, resizes it
/// with the antialiased-bilinear kernel to `(grid·PATCH_SIZE)` pixels, rescale+
/// normalizes, patchifies to `pixel_values` + `attention_mask`, and lifts the
/// base position grid to `position_embeddings`.
///
/// `rgb` is row-major RGB-interleaved (`width · height · 3`, as a validated
/// [`crate::embeddings::siglip::Rgb8Image`] guarantees); `base_grid` is the
/// parsed `POS_EMBED_ELEMS` base position grid.
///
/// # Errors
/// [`Error::PatchCount`] if the solved grid exceeds `budget` (a solver bug).
pub(crate) fn preprocess_image(
  rgb: &[u8],
  width: usize,
  height: usize,
  base_grid: &[f32],
  budget: usize,
) -> Result<VisionInputs, Error> {
  let (grid_h, grid_w) = fit_to_patch_budget(height, width, PATCH_SIZE, budget);

  // Decode to f32 [0, 255] in (H, W, C) row-major (the Rgb8Image layout).
  let img_f32: Vec<f32> = rgb.iter().map(|&b| b as f32).collect();

  // Resize (on 0..255 floats), then rescale+normalize — the processor's order.
  let dst_h = grid_h * PATCH_SIZE;
  let dst_w = grid_w * PATCH_SIZE;
  let mut resized = resize_bilinear_antialias(&img_f32, height, width, CHANNELS, dst_h, dst_w);
  for v in &mut resized {
    *v = normalize_pixel(*v);
  }

  let (pixel_values, attention_mask) = patchify(&resized, grid_h, grid_w, budget)?;
  let position_embeddings = lift_position_embeddings(base_grid, grid_h, grid_w, budget);

  Ok(VisionInputs {
    pixel_values,
    attention_mask,
    position_embeddings,
    grid: (grid_h, grid_w),
  })
}

#[cfg(test)]
mod tests;
