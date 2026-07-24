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
//! 2. [`resize_bilinear_antialias_u8`] — the PIL-parity antialiased-bilinear
//!    image resize in the **uint8 domain** (Pillow `Resample.c` fixed-point:
//!    22-bit coefficients, per-pass u8 rounding), delegated to [`colconv`]'s
//!    byte-exact q8 resampler and matching the SigLIP2 processor (resize on u8,
//!    THEN rescale+normalize). Its sibling f64 [`resize_bilinear_antialias`] is
//!    the position-embedding lift's kernel (stage 5), which the reference
//!    genuinely computes in float and which stays native here. The u8 path's
//!    bit-exactness holds for source axes ≤ [`MAX_IMAGE_AXIS`], which
//!    [`preprocess_image`] enforces.
//! 3. [`normalize_u8`] — rescale + normalize `((v/255) − 0.5)/0.5`.
//! 4. [`patchify`] — reshape the resized+normalized image to `[P, 768]` patch
//!    rows plus the `[P]` real/pad mask.
//! 5. [`lift_position_embeddings`] — resize the base `16×16×768` grid to the
//!    `(h_p, w_p)` grid, flatten row-major, and zero-pad to `[P, 768]`. Pad rows
//!    are zero-filled — a deliberate deviation from transformers (which fills
//!    them with the first resized row); pad positions are attention-masked, so
//!    the model output is exactly invariant to the fill (see the fn doc).
//!
//! The pixel-normalization constants and patch flatten order are pinned by
//! committed fixtures (`tests/siglip/fixtures/goldens/preprocess.json`, Wave B)
//! and the full-tensor parity against staged `.npy` fixtures (Wave C); the pure
//! stage math here is proven hermetically in `tests.rs`.
//!
//! The image resize is bit-exact fixed-point (uint8 taps, per-pass rounding), so
//! its `pixel_values` match the slow-processor fixture exactly (Wave B3/C2, no
//! tolerance). The position-embedding lift accumulates in `f64` for determinism,
//! with the measured-then-pinned tolerance (Wave B3) absorbing any residual
//! delta against the torch reference's working precision.

use crate::embeddings::siglip::{embedding::EMBEDDING_DIM, error::Error};

/// Vision patch side in pixels (a `16×16` patch). Architecture constant of the
/// `patch16` model — not a resolved parameter (unlike the patch budget `P`).
pub(crate) const PATCH_SIZE: usize = 16;

/// RGB channel count of a decoded image.
pub(crate) const CHANNELS: usize = 3;

/// Flattened per-patch dimension `CHANNELS · PATCH_SIZE · PATCH_SIZE = 3·16·16 =
/// 768` — the `pixel_values` inner dimension: a [`crate::embeddings::siglip::PreprocessedImage`]
/// carries `max_num_patches · PATCH_DIM` pixel values. Coincides with
/// [`EMBEDDING_DIM`] for `base-patch16`, but is a distinct quantity.
pub const PATCH_DIM: usize = CHANNELS * PATCH_SIZE * PATCH_SIZE;

/// Maximum accepted source-image extent per axis, `2²⁰ = 1 048 576` pixels.
/// `preprocess_image` rejects a larger axis with
/// [`Error::ImageDimensions`] before any resize work. The bound is a property
/// of the resize kernels, not of the model:
///
/// * **Pillow-exactness envelope.** Pillow passes the resize box through
///   `float box[4]` (`_imaging.c` parses `(ffff)`; `Resample.c`
///   `precompute_coeffs(int inSize, float in0, float in1, …)` computes
///   `scale = (double)(in1 - in0) / outSize`), so the source extent is rounded
///   to `f32` before the `f64` coefficient math. Every integer extent `≤ 2²⁴`
///   is exactly `f32`-representable, keeping both colconv's q8 image resampler
///   and the native `f64` `precompute_coeffs` (the pos-emb lift) bit-identical
///   to Pillow's; from `2²⁴ + 1` upward Pillow computes with a different extent
///   (`16 777 219 → 16 777 220.0f32`) and the u8 resize's bit-exact contract
///   would silently break. The cap keeps every accepted extent 16× inside the
///   envelope.
/// * **Bounded working memory.** The resize output (`dst_h · dst_w · 3` bytes)
///   and colconv's streaming working set both grow with the accepted axis — a
///   degenerate strip the budget solver otherwise accepts could demand
///   hundreds of MB. Under the cap the whole resize working set stays modest
///   even at the boundary.
///
/// `2²⁰` px per axis is far beyond any physically decodable image (record
/// stitched panoramas peak near `8.5 × 10⁵` px on the long axis), so the
/// bound rejects nothing realistic while making the Pillow-parity contract
/// hold over the entire accepted domain.
pub const MAX_IMAGE_AXIS: usize = 1 << 20;

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

/// Pixel rescale factor `1/255` as `f64` — bit-for-bit the checkpoint's
/// `preprocessor_config.json` `rescale_factor` literal `0.00392156862745098`
/// (the shortest round-trip repr of `1.0/255.0`). The reference multiplies the
/// `u8` pixel by this Python float (`f64`) and casts the product to `f32`
/// (`rescale(..., dtype=np.float32)`), so [`normalize_u8`] mirrors that order.
const RESCALE_FACTOR_F64: f64 = 1.0 / 255.0;
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
///
/// The sole live caller is the position-embedding lift
/// ([`lift_position_embeddings`], via [`resize_bilinear_antialias`]), whose
/// source is the fixed `16×16` base grid ([`POS_GRID_SIDE`]) — the image
/// resize path uses colconv's `u8` resampler instead
/// ([`resize_bilinear_antialias_u8`]). `16 ≪ 2²⁴`, so this pure-`f64` extent
/// math trivially coincides bit-for-bit with Pillow's `float box[4]` semantics
/// (see [`MAX_IMAGE_AXIS`]'s doc for the threshold).
///
/// # Errors
/// [`Error::PreprocessAllocation`] if a pathological source extent makes the
/// coefficient table's element count overflow `usize`, or a coefficient vector
/// cannot be reserved.
fn precompute_coeffs(in_size: usize, out_size: usize) -> Result<Vec<(usize, Vec<f64>)>, Error> {
  let scale = in_size as f64 / out_size as f64;
  let filterscale = if scale < 1.0 { 1.0 } else { scale };
  let support = filterscale; // triangle filter support is 1.0
  let inv_filterscale = 1.0 / filterscale;

  // Representability pre-guard: every clamped output draws at most
  // `2·⌈support⌉ + 1` taps, so the table holds at most
  // `out_size·(2·⌈support⌉ + 1)` weights. If that element count cannot even be
  // expressed in `usize` (a pathological source extent), no table could be
  // allocated — refuse with the size-overflow sentinel rather than aborting in
  // the fallible reservations below. Computed fully checked so the bound's own
  // arithmetic cannot overflow-abort.
  (support.ceil() as usize)
    .checked_mul(2)
    .and_then(|two_support| two_support.checked_add(1))
    .and_then(|ksize_bound| out_size.checked_mul(ksize_bound))
    .ok_or(Error::PreprocessAllocation { bytes: usize::MAX })?;

  let mut coeffs = Vec::new();
  coeffs
    .try_reserve_exact(out_size)
    .map_err(|_| Error::PreprocessAllocation {
      bytes: out_size.saturating_mul(core::mem::size_of::<(usize, Vec<f64>)>()),
    })?;
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

    let mut weights = Vec::new();
    weights
      .try_reserve_exact(taps)
      .map_err(|_| Error::PreprocessAllocation {
        bytes: taps.saturating_mul(core::mem::size_of::<f64>()),
      })?;
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
  Ok(coeffs)
}

/// Antialiased-bilinear resample of an `(src_h, src_w, channels)` row-major
/// `f32` buffer to `(dst_h, dst_w, channels)`. Separable: resize width, then
/// height (the PIL horizontal-then-vertical order), accumulating in `f64`.
///
/// This is the **position-embedding lift's** kernel (`channels = 768`), which
/// the reference computes in float (`F.interpolate` on the fp32 grid). The
/// image pixels take the uint8 kernel [`resize_bilinear_antialias_u8`] instead
/// (the SigLIP2 processor resizes on `u8`, then rescale+normalizes). Panics only
/// on an inconsistent `src` length (`src.len() != src_h·src_w·channels`) — a
/// caller-internal invariant.
///
/// # Errors
/// [`Error::PreprocessAllocation`] if a coefficient table cannot be sized or
/// reserved (a pathological axis extent). The pos-emb lift's `16×16` source
/// makes this unreachable in practice; the `Result` propagates the shared
/// kernel's allocation fallibility to its callers.
pub(crate) fn resize_bilinear_antialias(
  src: &[f32],
  src_h: usize,
  src_w: usize,
  channels: usize,
  dst_h: usize,
  dst_w: usize,
) -> Result<Vec<f32>, Error> {
  assert_eq!(
    src.len(),
    src_h * src_w * channels,
    "resize src length inconsistent with src_h·src_w·channels"
  );

  // Horizontal pass: (src_h, src_w, C) → (src_h, dst_w, C), kept in f64 to avoid
  // an intermediate rounding between the two separable passes.
  let wc = precompute_coeffs(src_w, dst_w)?;
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
  let hc = precompute_coeffs(src_h, dst_h)?;
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
  Ok(out)
}

/// PIL-parity antialiased-bilinear resize of a packed RGB8 image
/// (`(src_h, src_w, 3)` row-major, RGB-interleaved) to `(dst_h, dst_w, 3)`,
/// delegated to [`colconv`]'s fixed-point q8 resampler.
///
/// colconv's `Triangle` (PIL BILINEAR) u8 path reproduces Pillow's 8bpc
/// `Resample.c` pipeline byte-for-byte — 22-bit coefficients, per-pass `u8`
/// rounding of the horizontal intermediate — which is exactly the SigLIP2
/// processor's image-resize semantics (resize on `u8`, THEN rescale+normalize;
/// the per-pass rounding is part of the reference and is what a pure-`f64`
/// resize cannot emulate). The byte-exactness is enforced upstream by colconv's
/// tolerance-0 Pillow golden suite and downstream by this module's committed E3
/// oracles (`tests.rs`).
///
/// The only caller, [`preprocess_image`], bounds `src_h`/`src_w` by
/// [`MAX_IMAGE_AXIS`], which keeps every accepted extent exactly
/// `f32`-representable — inside Pillow's `float box[4]` envelope, so colconv's
/// coefficient math matches Pillow's (see [`MAX_IMAGE_AXIS`]) — and keeps
/// colconv's streaming working set small.
///
/// # Errors
/// [`Error::PreprocessAllocation`] if the output buffer cannot be reserved, a
/// dimension does not fit colconv's `u32` frame geometry, or colconv rejects the
/// resize plan — returned instead of aborting the process (`colconv` reserves
/// fallibly and [`Rgb24Frame::try_new`](colconv::frame::Rgb24Frame::try_new)
/// validates the geometry rather than panicking). `bytes` is the `dst_h · dst_w
/// · 3` output-buffer size — representable even when its reservation or
/// colconv's own resize plan is what actually failed — or [`usize::MAX`] for a
/// geometry that could never be represented.
pub(crate) fn resize_bilinear_antialias_u8(
  src: &[u8],
  src_h: usize,
  src_w: usize,
  dst_h: usize,
  dst_w: usize,
) -> Result<Vec<u8>, Error> {
  use colconv::{Convert, frame::Rgb24Frame, resample::Triangle};

  // colconv's frame geometry is `u32`; a source extent that does not fit could
  // never be represented, let alone allocated. Only reachable through a direct
  // pathological-geometry call — `preprocess_image` caps both axes by
  // `MAX_IMAGE_AXIS`, far inside `u32`.
  let width =
    u32::try_from(src_w).map_err(|_| Error::PreprocessAllocation { bytes: usize::MAX })?;
  let height =
    u32::try_from(src_h).map_err(|_| Error::PreprocessAllocation { bytes: usize::MAX })?;
  let stride = width
    .checked_mul(CHANNELS as u32)
    .ok_or(Error::PreprocessAllocation { bytes: usize::MAX })?;

  // Output buffer (`dst_h · dst_w · 3` bytes), reserved fallibly so a
  // pathological target geometry returns a typed error instead of aborting on
  // allocator failure.
  let dst_len = dst_w
    .checked_mul(dst_h)
    .and_then(|hw| hw.checked_mul(CHANNELS))
    .ok_or(Error::PreprocessAllocation { bytes: usize::MAX })?;
  let mut out = Vec::new();
  out
    .try_reserve_exact(dst_len)
    .map_err(|_| Error::PreprocessAllocation { bytes: dst_len })?;
  out.resize(dst_len, 0);

  // The tightly-packed RGB24 source frame; `try_new` (not the panicking `new`)
  // validates dimensions and plane length, returning a typed error on a
  // pathological direct-call extent.
  let frame = Rgb24Frame::try_new(src, width, height, stride)
    .map_err(|_| Error::PreprocessAllocation { bytes: usize::MAX })?;

  // Filtered (windowed-triangle = PIL BILINEAR) resample, any ratio; the sole
  // fallible call assembles the sink and walks the frame into `out`. `out` is
  // already sized to `dst_len` (a representable value) at this point, so a
  // colconv-side plan rejection or internal reservation failure here is
  // reported at that real size — not aliased onto the `usize::MAX` sentinel,
  // which is reserved for a geometry that could not be represented at all.
  Convert::from(&frame)
    .resize_with(Triangle, dst_w, dst_h)
    .rgb(&mut out)
    .run()
    .map_err(|_| Error::PreprocessAllocation { bytes: dst_len })?;

  Ok(out)
}

/// Rescale + normalize one `u8` pixel channel to `((v/255) − 0.5)/0.5 ∈ [−1, 1]`
/// — the SigLIP2 processor's `do_rescale` + `do_normalize` (mean `0.5`, std
/// `0.5`). Mirrors the reference dtype order: multiply the `u8` by the `f64`
/// rescale factor, cast the product to `f32`, then normalize in `f32`.
pub(crate) fn normalize_u8(v: u8) -> f32 {
  ((f64::from(v) * RESCALE_FACTOR_F64) as f32 - IMAGE_MEAN) / IMAGE_STD
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
/// grid with the f64 antialiased-bilinear kernel, take its row-major
/// `(grid_h·grid_w, 768)` flattening (the resize already emits that layout), and
/// zero-pad the remaining `budget − grid_h·grid_w` rows.
///
/// Pad rows are **zero-filled — a deliberate deviation** from transformers
/// `resize_positional_embeddings`, which fills them with the first resized row
/// (an artifact of its `torch.empty` initialization). Pad positions are
/// attention-masked in the encoder and the pooling head (additive `−1e4` →
/// exactly zero softmax weight), so the model output is exactly invariant to the
/// fill; zero is chosen because it is validatable and matches the pixel-pad
/// convention. Wave C compares the lifted real rows against the reference and
/// asserts the pad rows zero.
///
/// `base_grid` must be `POS_EMBED_ELEMS` long (as [`parse_base_pos_grid`]
/// returns) and `grid_h · grid_w ≤ budget` (the budget solver guarantees it;
/// excess is defensively truncated rather than panicking).
///
/// # Errors
/// [`Error::PreprocessAllocation`] if the resize's coefficient tables cannot be
/// sized or reserved (propagated from [`resize_bilinear_antialias`]; the fixed
/// `16×16` base grid makes this unreachable in practice).
pub(crate) fn lift_position_embeddings(
  base_grid: &[f32],
  grid_h: usize,
  grid_w: usize,
  budget: usize,
) -> Result<Vec<f32>, Error> {
  let resized = resize_bilinear_antialias(
    base_grid,
    POS_GRID_SIDE,
    POS_GRID_SIDE,
    EMBEDDING_DIM,
    grid_h,
    grid_w,
  )?;
  let mut out = vec![0.0f32; budget * EMBEDDING_DIM];
  let copy_len = resized.len().min(out.len());
  out[..copy_len].copy_from_slice(&resized[..copy_len]);
  Ok(out)
}

/// The three vision-graph input tensors produced from one decoded image, plus
/// the resolved `(grid_h, grid_w)` patch grid.
#[derive(Debug)]
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
/// with the uint8 PIL-parity antialiased-bilinear kernel to `(grid·PATCH_SIZE)`
/// pixels, rescale+normalizes, patchifies to `pixel_values` + `attention_mask`,
/// and lifts the base position grid to `position_embeddings`.
///
/// `rgb` is row-major RGB-interleaved (`width · height · 3`, as a validated
/// [`crate::embeddings::siglip::Rgb8Image`] guarantees); `base_grid` is the
/// parsed `POS_EMBED_ELEMS` base position grid.
///
/// # Errors
/// [`Error::ImageDimensions`] if an image axis exceeds [`MAX_IMAGE_AXIS`];
/// [`Error::PatchCount`] if the solved grid exceeds `budget` (a solver bug);
/// [`Error::PreprocessAllocation`] if a resize working buffer cannot be sized or
/// reserved (pathological source geometry).
pub(crate) fn preprocess_image(
  rgb: &[u8],
  width: usize,
  height: usize,
  base_grid: &[f32],
  budget: usize,
) -> Result<VisionInputs, Error> {
  if width > MAX_IMAGE_AXIS || height > MAX_IMAGE_AXIS {
    return Err(Error::ImageDimensions { width, height });
  }

  let (grid_h, grid_w) = fit_to_patch_budget(height, width, PATCH_SIZE, budget);

  // Resize in the u8 domain (colconv's PIL-parity q8 resampler), then
  // rescale+normalize to f32 — the SigLIP2 processor's order AND dtype (it
  // resizes the u8 image, rounding each pass to u8, then casts to f32). No
  // whole-image f32 buffer is materialized; colconv streams the resize and the
  // only owned buffer is its fallibly-reserved u8 output.
  let dst_h = grid_h * PATCH_SIZE;
  let dst_w = grid_w * PATCH_SIZE;
  let resized_u8 = resize_bilinear_antialias_u8(rgb, height, width, dst_h, dst_w)?;
  let resized: Vec<f32> = resized_u8.iter().map(|&v| normalize_u8(v)).collect();

  let (pixel_values, attention_mask) = patchify(&resized, grid_h, grid_w, budget)?;
  let position_embeddings = lift_position_embeddings(base_grid, grid_h, grid_w, budget)?;

  Ok(VisionInputs {
    pixel_values,
    attention_mask,
    position_embeddings,
    grid: (grid_h, grid_w),
  })
}

#[cfg(test)]
mod tests;
