use super::*;

const P: usize = 512;

// ── A6: budget solver ────────────────────────────────────────────────────────

/// The one fully-known real oracle pair: the probe's 320×240 image
/// (`spatial_shapes = (19, 26)` at the 512 budget). `320×240` is W×H, so
/// `image_height = 240`, `image_width = 320`.
#[test]
fn budget_solver_matches_known_320x240_oracle() {
  assert_eq!(fit_to_patch_budget(240, 320, PATCH_SIZE, P), (19, 26));
}

/// A square image scales uniformly to the largest square grid fitting the
/// budget: `⌊√512⌋ = 22` (`22² = 484 ≤ 512`, `23² = 529 > 512`).
#[test]
fn budget_solver_square_image_fills_the_square_grid() {
  assert_eq!(fit_to_patch_budget(512, 512, PATCH_SIZE, P), (22, 22));
  // Absolute size is (near-)irrelevant — a smaller square lands on the same grid.
  assert_eq!(fit_to_patch_budget(64, 64, PATCH_SIZE, P), (22, 22));
}

/// The probe's eight measured `aspect → spatial_shapes` rows (P = 512), the real
/// oracle already recorded in the conversion probe. Feeding representative
/// dimensions at each aspect (long side 640, the COCO norm) reproduces every
/// measured `(h_p, w_p)` grid EXACTLY — a cross-check of the port against
/// measured NaFlex truth. `aspect = width / height`; a portrait (`< 1`) has the
/// long side on height, a landscape (`≥ 1`) on width.
#[test]
fn budget_solver_reproduces_probe_aspect_table() {
  // (image_height, image_width, expected grid_h, grid_w)
  let rows: &[(usize, usize, usize, usize)] = &[
    (640, 425, 28, 18), // aspect 0.664
    (640, 480, 26, 19), // aspect 0.750
    (640, 586, 23, 22), // aspect 0.916
    (483, 640, 19, 26), // aspect 1.325
    (480, 640, 19, 26), // aspect 1.333
    (427, 640, 18, 27), // aspect 1.499
    (426, 640, 18, 28), // aspect 1.502
  ];
  for &(h, w, eh, ew) in rows {
    assert_eq!(
      fit_to_patch_budget(h, w, PATCH_SIZE, P),
      (eh, ew),
      "budget solver diverged from the probe oracle for {h}×{w}"
    );
  }
}

/// Budget + maximality invariants across a spread of shapes: the grid always
/// fits (`h_p·w_p ≤ P`), never degenerates (`≥ 1` per axis), and is maximal —
/// growing BOTH axes by one patch would exceed the budget.
#[test]
fn budget_solver_respects_budget_and_is_maximal() {
  for &(h, w) in &[
    (240, 320),
    (640, 425),
    (512, 512),
    (100, 3000),
    (1000, 1000),
    (37, 91),
    (640, 587),
  ] {
    let (hp, wp) = fit_to_patch_budget(h, w, PATCH_SIZE, P);
    assert!(hp >= 1 && wp >= 1, "{h}×{w} degenerate grid ({hp},{wp})");
    assert!(hp * wp <= P, "{h}×{w} exceeds budget: {hp}·{wp}");
    assert!(
      (hp + 1) * (wp + 1) > P,
      "{h}×{w} not maximal: ({}+1)·({}+1) ≤ {P}",
      hp,
      wp
    );
  }
}

/// The solver is (near-)scale-invariant: the same aspect ratio at different
/// absolute sizes lands on the same grid (the probe measured the small 320×240
/// and a full-size 1.333 image onto the identical `(19, 26)`).
#[test]
fn budget_solver_is_scale_invariant_for_a_fixed_aspect() {
  let g = fit_to_patch_budget(240, 320, PATCH_SIZE, P);
  assert_eq!(g, (19, 26));
  assert_eq!(fit_to_patch_budget(480, 640, PATCH_SIZE, P), g);
  assert_eq!(fit_to_patch_budget(120, 160, PATCH_SIZE, P), g);
}

/// An extreme aspect ratio clamps the short side to a single patch (the
/// `max(patch, …)` floor) while the long side absorbs the budget.
#[test]
fn budget_solver_extreme_aspect_clamps_short_side_to_one_patch() {
  let (hp, wp) = fit_to_patch_budget(16, 16_000, PATCH_SIZE, P);
  assert_eq!(hp, 1, "the 1-patch short-side clamp must hold");
  assert!(wp >= 1 && hp * wp <= P, "grid ({hp},{wp}) invalid");
}

// ── A8: antialiased-bilinear resize kernel ───────────────────────────────────

/// Identity resize (`out == in`) is exact per element and per channel — the
/// coefficient at each output is a single unit weight.
#[test]
fn resize_identity_is_exact() {
  // 2×2, three channels, distinct values.
  let src: Vec<f32> = (0..2 * 2 * 3).map(|i| i as f32).collect();
  let out = resize_bilinear_antialias(&src, 2, 2, 3, 2, 2).expect("identity resize");
  assert_eq!(out, src);
}

/// A constant field stays constant under any resize (weights sum to 1).
#[test]
fn resize_of_constant_field_is_constant() {
  let src = vec![0.375f32; 3 * 5 * 2]; // 3×5, 2 channels
  let up = resize_bilinear_antialias(&src, 3, 5, 2, 7, 9).expect("upscale");
  assert!(
    up.iter().all(|&v| (v - 0.375).abs() <= 1e-6),
    "upscale drifted"
  );
  let down = resize_bilinear_antialias(&src, 3, 5, 2, 2, 2).expect("downscale");
  assert!(
    down.iter().all(|&v| (v - 0.375).abs() <= 1e-6),
    "downscale drifted"
  );
}

/// Upscale `[0, 1]` (2→4) matches the hand-computed align-corners=false bilinear
/// `[0, 0.25, 0.75, 1.0]` EXACTLY (edge replication at the ends, quarter steps
/// interior).
#[test]
fn resize_upscale_matches_hand_computed_bilinear() {
  let out = resize_bilinear_antialias(&[0.0, 1.0], 1, 2, 1, 1, 4).expect("upscale resize");
  assert_eq!(out, vec![0.0, 0.25, 0.75, 1.0]);
}

/// A 2×2 checker upscaled to 4×4 matches the hand-computed separable bilinear
/// surface: corners replicate the source, the interior 2×2 block is the
/// symmetric `{0.375, 0.625}` blend.
#[test]
fn resize_checker_upscale_matches_hand_computed_surface() {
  // [[0,1],[1,0]] row-major, single channel.
  let src = [0.0f32, 1.0, 1.0, 0.0];
  let out = resize_bilinear_antialias(&src, 2, 2, 1, 4, 4).expect("checker upscale");
  let at = |r: usize, c: usize| out[r * 4 + c];
  // Corners replicate.
  assert_eq!(
    (at(0, 0), at(0, 3), at(3, 0), at(3, 3)),
    (0.0, 1.0, 1.0, 0.0)
  );
  // Interior 2×2 block.
  for (r, c, want) in [
    (1, 1, 0.375f32),
    (1, 2, 0.625),
    (2, 1, 0.625),
    (2, 2, 0.375),
  ] {
    assert!(
      (at(r, c) - want).abs() <= 1e-6,
      "checker[{r}][{c}] = {}",
      at(r, c)
    );
  }
}

/// Antialiased downscale of a step edge `[0,0,255,255]` (4→2) low-passes
/// symmetrically: the two outputs are mirror images summing to the full range
/// (`v` and `255 − v`), and monotonic — proving the triangle support widened
/// (a non-antialiased 2-tap bilinear would just subsample to `[0, 255]`).
#[test]
fn resize_antialias_downscale_is_symmetric_lowpass() {
  let out = resize_bilinear_antialias(&[0.0, 0.0, 255.0, 255.0], 1, 4, 1, 1, 2)
    .expect("antialias downscale");
  assert!(out[0] < out[1], "must be monotonic increasing");
  assert!(
    (out[0] + out[1] - 255.0).abs() <= 1e-3,
    "must be symmetric around 127.5"
  );
  // The edge is smoothed (not a hard subsample to 0 / 255).
  assert!(
    out[0] > 1.0 && out[1] < 254.0,
    "antialias must low-pass the edge"
  );
}

/// Channels are resampled independently — one channel's values never bleed into
/// another. Two RGB pixels `[(0,10,20),(40,50,60)]` upscaled 2→4 give each
/// channel its own 1D upscale (`[v0, .75v0+.25v1, .25v0+.75v1, v1]`).
#[test]
fn resize_keeps_channels_independent() {
  let src = [0.0f32, 10.0, 20.0, 40.0, 50.0, 60.0]; // 1×2, RGB
  let out = resize_bilinear_antialias(&src, 1, 2, 3, 1, 4).expect("channel resize");
  // Reconstruct expected per channel.
  let up1d = |a: f32, b: f32| [a, 0.75 * a + 0.25 * b, 0.25 * a + 0.75 * b, b];
  for c in 0..3 {
    let want = up1d(src[c], src[3 + c]);
    for x in 0..4 {
      assert!(
        (out[x * 3 + c] - want[x]).abs() <= 1e-6,
        "channel {c} pixel {x}: {} != {}",
        out[x * 3 + c],
        want[x]
      );
    }
  }
}

/// Separability sanity: a vertically-constant image (identical rows) stays
/// vertically constant after a height resize, since each column is constant.
#[test]
fn resize_preserves_vertical_constancy() {
  // 4×2×1 with all rows == [3.0, 7.0].
  let src = [3.0f32, 7.0, 3.0, 7.0, 3.0, 7.0, 3.0, 7.0];
  let out = resize_bilinear_antialias(&src, 4, 2, 1, 2, 2).expect("height resize"); // 2×2
  assert!((out[0] - out[2]).abs() <= 1e-6, "column 0 not constant");
  assert!((out[1] - out[3]).abs() <= 1e-6, "column 1 not constant");
}

// ── E3: uint8 PIL-parity resize kernel (fixed-point oracles) ──────────────────

/// Identity dims return the input bytes exactly: each output is a single
/// unit-weight tap, and the `1 << 21` offset truncates back to the input value.
#[test]
fn resize_u8_identity_is_exact() {
  let src = [17u8, 200, 3, 250, 9, 128, 44, 71, 255]; // 3×3, 1 channel
  let out = resize_bilinear_antialias_u8(&src, 3, 3, 1, 3, 3).expect("resize");
  assert_eq!(out, src);
}

/// Any constant field stays exactly constant under up- and down-scale: the
/// quantized weight sum is within `ksize/2` of `2²²`, so `c·δ < 2²¹` for this
/// test's geometries (the margin holds while `ksize < 16448`) and the offset
/// truncates back to `c`.
#[test]
fn resize_u8_constant_field_is_constant() {
  let src = vec![123u8; 5 * 7 * 3]; // 5×7, 3 channels
  let up = resize_bilinear_antialias_u8(&src, 5, 7, 3, 9, 4).expect("upscale");
  assert!(
    up.iter().all(|&v| v == 123),
    "upscale drifted from constant"
  );
  let down = resize_bilinear_antialias_u8(&src, 5, 7, 3, 2, 2).expect("downscale");
  assert!(
    down.iter().all(|&v| v == 123),
    "downscale drifted from constant"
  );
}

/// A 2×2 checker `[[0,255],[255,0]]` upscaled to 4×4 matches the hand-derived
/// Pillow 12.3.0 fixed-point grid EXACTLY (2→4 weights `[1]`, `[.75,.25]`,
/// `[.25,.75]`, `[1]`; quantized `[4194304]`, `[3145728,1048576]`, …).
#[test]
fn resize_u8_matches_hand_computed_pil_grid_checker() {
  let src = [0u8, 255, 255, 0];
  let out = resize_bilinear_antialias_u8(&src, 2, 2, 1, 4, 4).expect("resize");
  #[rustfmt::skip]
  let expected: [u8; 16] = [
      0,  64, 191, 255,
     64,  96, 159, 191,
    191, 159,  96,  64,
    255, 191,  64,   0,
  ];
  assert_eq!(out, expected);
}

/// A 2×2 `[[0,1],[255,255]]` upscaled to 4×4 matches the hand-derived grid, with
/// the discriminant cells `[1][2] = 65` and `[2][2] = 192`. A float pipeline
/// rounded once at the end yields 64 and 191 there (`0.75·0.75 + 0.25·255 =
/// 64.3125`; `0.25·0.75 + 0.75·255 = 191.4375`), so these cells pin the uint8
/// per-pass (u8 intermediate) rounding as non-vacuous.
#[test]
fn resize_u8_per_pass_rounding_discriminates_from_float() {
  let src = [0u8, 1, 255, 255];
  let out = resize_bilinear_antialias_u8(&src, 2, 2, 1, 4, 4).expect("resize");
  #[rustfmt::skip]
  let expected: [u8; 16] = [
      0,   0,   1,   1,
     64,  64,  65,  65,
    191, 191, 192, 192,
    255, 255, 255, 255,
  ];
  assert_eq!(out, expected);
  // Cells [1][2] and [2][2] (indices 6 and 10) are the float-vs-uint8 tell.
  assert_eq!((out[6], out[10]), (65, 192));
}

/// Pathologically large source dims return a typed
/// [`Error::PreprocessAllocation`] from [`checked_len3`]'s `checked_mul`
/// working-buffer sizing arm — no panic, no abort. (The `try_reserve` arm is
/// not portably forcible.)
#[test]
fn resize_u8_rejects_overflowing_geometry() {
  match resize_bilinear_antialias_u8(&[0u8; 12], usize::MAX / 2, 2, 3, 2, 2) {
    Err(Error::PreprocessAllocation { bytes }) => assert_eq!(bytes, usize::MAX),
    other => panic!("expected PreprocessAllocation, got {other:?}"),
  }
}

/// A pathologically large *source* extent trips the coefficient-table
/// representability pre-guard inside [`precompute_coeffs`] (the horizontal
/// axis' `out·(2⌈support⌉+1)` element count overflows `usize`), returning a
/// typed [`Error::PreprocessAllocation`] before any tap is indexed — no panic,
/// no abort. This is the coefficient-table arm, distinct from the working-buffer
/// [`checked_len3`] arm above. (The `try_reserve` arms are not portably
/// forcible.)
#[test]
fn resize_u8_rejects_coeff_table_overflow() {
  // src_w = usize::MAX/2, dst_w = 16 ⇒ support ≈ 2⁵⁹, so 2⌈support⌉+1 = 2⁶⁰+1
  // and 16·(2⁶⁰+1) overflows usize. precompute_coeffs(src_w, dst_w) is the u8
  // kernel's first action, so the guard fires before any buffer or src indexing;
  // the tiny src is never read.
  match resize_bilinear_antialias_u8(&[0u8; 12], 2, usize::MAX / 2, 3, 2, 16) {
    Err(Error::PreprocessAllocation { bytes }) => assert_eq!(bytes, usize::MAX),
    other => panic!("expected PreprocessAllocation, got {other:?}"),
  }
}

/// `preprocess_image` routes pixels through the uint8 kernel + [`normalize_u8`],
/// NOT the float path. Budget 1 makes the whole image a single 16×16 patch, so
/// the real patch row equals `normalize_u8` of the u8-resized bytes exactly; and
/// the u8 resize provably differs from a float-then-round resize for this source
/// (the per-pass rounding), so the routing is observable, not vacuous.
#[test]
fn preprocess_image_pixel_values_come_from_u8_resize() {
  // src pattern [[0,1],[255,255]] replicated on all 3 channels — this diverges
  // from a float resize under 2→16 (the per-pass u8 rounding).
  let pattern = [0u8, 1, 255, 255];
  let mut rgb = vec![0u8; 2 * 2 * CHANNELS];
  for (px, &v) in pattern.iter().enumerate() {
    for c in 0..CHANNELS {
      rgb[px * CHANNELS + c] = v;
    }
  }
  let base = vec![0.0f32; POS_EMBED_ELEMS];
  let out = preprocess_image(&rgb, 2, 2, &base, 1).expect("preprocess");
  assert_eq!(out.grid, (1, 1), "budget 1 → single-patch grid");

  // The single real patch is normalize_u8 of the u8-resized image, exactly.
  let resized_u8 = resize_bilinear_antialias_u8(&rgb, 2, 2, CHANNELS, 16, 16).expect("resize");
  assert_eq!(resized_u8.len(), 16 * 16 * CHANNELS);
  for (k, &v) in resized_u8.iter().enumerate() {
    assert_eq!(
      out.pixel_values[k],
      normalize_u8(v),
      "pixel_values[{k}] must be normalize_u8 of the u8-resized byte"
    );
  }

  // Non-vacuity: a float resize rounded to u8 diverges from the per-pass u8
  // kernel for this source, so preprocess_image is provably off the old float
  // path.
  let rgb_f32: Vec<f32> = rgb.iter().map(|&b| f32::from(b)).collect();
  let float_resized =
    resize_bilinear_antialias(&rgb_f32, 2, 2, CHANNELS, 16, 16).expect("float resize");
  let float_u8: Vec<u8> = float_resized
    .iter()
    .map(|&v| v.round().clamp(0.0, 255.0) as u8)
    .collect();
  assert!(
    resized_u8.iter().zip(&float_u8).any(|(&u, &f)| u != f),
    "uint8 per-pass resize must differ from a float-then-round resize here"
  );
}

// ── A7: normalize + patchify + mask ──────────────────────────────────────────

/// The rescale+normalize maps the u8 range `[0, 255]` onto `[-1, 1]` at the
/// pinned constants `((v/255) − 0.5)/0.5`: the endpoints are exact and the
/// midpoint is near zero.
#[test]
fn normalize_u8_maps_range_to_unit_interval() {
  assert_eq!(normalize_u8(0), -1.0);
  assert_eq!(normalize_u8(255), 1.0);
  assert!(normalize_u8(128).abs() < 0.01);
}

/// Patchify places each pixel at the exact `(patch_row, patch_col)` × `(py, px,
/// channel)` slot: a synthetic image whose every pixel channel encodes its own
/// `(y, x, c)` coordinate must land where the reshape says. A transpose or
/// off-by-one in the flatten order relocates values and reds this.
#[test]
fn patchify_places_pixels_at_exact_slots() {
  let grid_h = 2;
  let grid_w = 3;
  let img_h = grid_h * PATCH_SIZE;
  let img_w = grid_w * PATCH_SIZE;
  // Distinguishable encoding: value = (y*10000 + x*10 + c) as f32.
  let mut img = vec![0.0f32; img_h * img_w * CHANNELS];
  for y in 0..img_h {
    for x in 0..img_w {
      for c in 0..CHANNELS {
        img[(y * img_w + x) * CHANNELS + c] = (y * 10_000 + x * 10 + c) as f32;
      }
    }
  }
  let budget = 10;
  let (pixel_values, mask) = patchify(&img, grid_h, grid_w, budget).expect("patchify");
  assert_eq!(pixel_values.len(), budget * PATCH_DIM);
  assert_eq!(mask.len(), budget);

  // Verify the full reshape by re-deriving each slot's expected source pixel.
  for ph in 0..grid_h {
    for pw in 0..grid_w {
      let row = ph * grid_w + pw;
      let mut k = 0;
      for py in 0..PATCH_SIZE {
        for px in 0..PATCH_SIZE {
          for c in 0..CHANNELS {
            let y = ph * PATCH_SIZE + py;
            let x = pw * PATCH_SIZE + px;
            let want = (y * 10_000 + x * 10 + c) as f32;
            assert_eq!(
              pixel_values[row * PATCH_DIM + k],
              want,
              "slot (ph={ph},pw={pw},py={py},px={px},c={c}) misplaced"
            );
            k += 1;
          }
        }
      }
    }
  }
}

/// The mask marks exactly the `grid_h·grid_w` real patches `1.0` and zero-pads
/// the rest; the padded `pixel_values` rows are bitwise zero.
#[test]
fn patchify_mask_and_padding_are_correct() {
  let grid_h = 2;
  let grid_w = 3; // 6 real patches
  let img = vec![0.5f32; (grid_h * PATCH_SIZE) * (grid_w * PATCH_SIZE) * CHANNELS];
  let budget = 8;
  let (pixel_values, mask) = patchify(&img, grid_h, grid_w, budget).expect("patchify");

  let n_real = grid_h * grid_w;
  assert_eq!(
    mask.iter().sum::<f32>(),
    n_real as f32,
    "mask must count real patches"
  );
  assert!(
    mask[..n_real].iter().all(|&m| m == 1.0),
    "real patches masked 1.0"
  );
  assert!(
    mask[n_real..].iter().all(|&m| m == 0.0),
    "pad patches masked 0.0"
  );
  // Pad rows are bitwise zero.
  assert!(
    pixel_values[n_real * PATCH_DIM..].iter().all(|&v| v == 0.0),
    "padded pixel rows must be zero"
  );
}

/// Patchify defensively rejects a grid larger than the budget with a typed
/// error (never an out-of-bounds write).
#[test]
fn patchify_rejects_grid_over_budget() {
  let grid_h = 3;
  let grid_w = 3; // 9 patches
  let img = vec![0.0f32; (grid_h * PATCH_SIZE) * (grid_w * PATCH_SIZE) * CHANNELS];
  match patchify(&img, grid_h, grid_w, 8) {
    Err(Error::PatchCount { got: 9, max: 8 }) => {}
    other => panic!("expected PatchCount, got {other:?}"),
  }
}

// ── A9: position-embedding grid (parse + lift + pad) ──────────────────────────

/// The raw sidecar length is hard-validated to the exact `16·16·768·4` byte
/// count; a short/long file is rejected, a correct one parses to
/// `POS_EMBED_ELEMS` little-endian f32.
#[test]
fn parse_base_pos_grid_validates_exact_byte_length() {
  assert_eq!(POS_EMBED_BYTES, 16 * 16 * 768 * 4);
  assert_eq!(POS_EMBED_BYTES, 786_432);

  let good = vec![0u8; POS_EMBED_BYTES];
  let grid = parse_base_pos_grid(&good).expect("exact length parses");
  assert_eq!(grid.len(), POS_EMBED_ELEMS);

  match parse_base_pos_grid(&[0u8; 16]) {
    Err(Error::PosEmbedLength { got: 16, expected }) => assert_eq!(expected, POS_EMBED_BYTES),
    other => panic!("expected PosEmbedLength, got {other:?}"),
  }
  let long = vec![0u8; POS_EMBED_BYTES + 4];
  assert!(matches!(
    parse_base_pos_grid(&long),
    Err(Error::PosEmbedLength { .. })
  ));
}

/// Round-trip: a known little-endian f32 pattern parses back to those floats.
#[test]
fn parse_base_pos_grid_decodes_little_endian_f32() {
  let mut bytes = vec![0u8; POS_EMBED_BYTES];
  bytes[0..4].copy_from_slice(&1.5f32.to_le_bytes());
  bytes[4..8].copy_from_slice(&(-2.25f32).to_le_bytes());
  let grid = parse_base_pos_grid(&bytes).expect("parse");
  assert_eq!(grid[0], 1.5);
  assert_eq!(grid[1], -2.25);
}

/// The lift resizes the base grid to the patch grid, flattens row-major, and
/// zero-pads to `[budget, EMBEDDING_DIM]`. A constant base grid resizes to the
/// same constant on every real patch row, with the pad rows bitwise zero — this
/// pins the flatten + pad plumbing (the resize kernel itself is covered above).
#[test]
fn lift_position_embeddings_flattens_and_zero_pads() {
  let base = vec![0.7f32; POS_EMBED_ELEMS]; // constant grid
  let grid_h = 3;
  let grid_w = 4; // 12 real patches
  let budget = 20;
  let lifted = lift_position_embeddings(&base, grid_h, grid_w, budget).expect("lift");
  assert_eq!(lifted.len(), budget * EMBEDDING_DIM);

  let n_real = grid_h * grid_w;
  // Real rows: resize of a constant grid is the same constant.
  assert!(
    lifted[..n_real * EMBEDDING_DIM]
      .iter()
      .all(|&v| (v - 0.7).abs() <= 1e-6),
    "real position rows must carry the resized (constant) grid"
  );
  // Pad rows: bitwise zero.
  assert!(
    lifted[n_real * EMBEDDING_DIM..].iter().all(|&v| v == 0.0),
    "padded position rows must be zero"
  );
}

/// The lift's per-patch row layout matches patchify's: distinct base-grid rows
/// land on distinct position rows in `(grid_row, grid_col)` order. An identity
/// resize (`grid == POS_GRID_SIDE`) makes the mapping exact, so a per-row
/// signature proves alignment with the patch order.
#[test]
fn lift_position_embeddings_row_order_matches_patch_order() {
  // Give each base-grid cell a per-cell signature in channel 0.
  let mut base = vec![0.0f32; POS_EMBED_ELEMS];
  for gy in 0..POS_GRID_SIDE {
    for gx in 0..POS_GRID_SIDE {
      base[(gy * POS_GRID_SIDE + gx) * EMBEDDING_DIM] = (gy * 100 + gx) as f32;
    }
  }
  // Identity grid (16×16) → resize is exact, one position row per cell.
  let budget = POS_GRID_SIDE * POS_GRID_SIDE + 5;
  let lifted = lift_position_embeddings(&base, POS_GRID_SIDE, POS_GRID_SIDE, budget).expect("lift");
  for gy in 0..POS_GRID_SIDE {
    for gx in 0..POS_GRID_SIDE {
      let row = gy * POS_GRID_SIDE + gx;
      assert_eq!(
        lifted[row * EMBEDDING_DIM],
        (gy * 100 + gx) as f32,
        "position row {row} misaligned with patch order"
      );
    }
  }
}

// ── A6–A10: full model-free image pipeline (preprocess_image) ─────────────────

/// A synthetic decoded RGB image (row-major, RGB-interleaved u8) with a
/// deterministic gradient.
fn synthetic_rgb(width: usize, height: usize) -> Vec<u8> {
  let mut data = vec![0u8; width * height * CHANNELS];
  for y in 0..height {
    for x in 0..width {
      let base = (y * width + x) * CHANNELS;
      data[base] = ((x * 255) / width.max(1)) as u8;
      data[base + 1] = ((y * 255) / height.max(1)) as u8;
      data[base + 2] = ((x + y) % 256) as u8;
    }
  }
  data
}

/// The full pipeline yields the three graph tensors at the budget's shapes, with
/// the mask counting exactly the solved grid's real patches and normalized
/// pixels inside `[-1, 1]`.
#[test]
fn preprocess_image_produces_budget_shaped_tensors() {
  let (w, h) = (320usize, 240usize);
  let base = vec![0.1f32; POS_EMBED_ELEMS];
  let rgb = synthetic_rgb(w, h);
  let out = preprocess_image(&rgb, w, h, &base, P).expect("preprocess");

  assert_eq!(out.grid, (19, 26), "grid must match the 320×240 oracle");
  assert_eq!(out.pixel_values.len(), P * PATCH_DIM);
  assert_eq!(out.attention_mask.len(), P);
  assert_eq!(out.position_embeddings.len(), P * EMBEDDING_DIM);

  let n_real = out.grid.0 * out.grid.1;
  assert_eq!(out.attention_mask.iter().sum::<f32>(), n_real as f32);
  assert!(
    out.pixel_values.iter().all(|&v| (-1.0..=1.0).contains(&v)),
    "normalized pixels must lie in [-1, 1]"
  );
  // Pad regions of both pixel_values and position_embeddings are zero.
  assert!(
    out.pixel_values[n_real * PATCH_DIM..]
      .iter()
      .all(|&v| v == 0.0)
  );
  assert!(
    out.position_embeddings[n_real * EMBEDDING_DIM..]
      .iter()
      .all(|&v| v == 0.0)
  );
}

/// The pipeline is deterministic — the same image yields byte-identical tensors
/// across runs (the hermetic determinism evidence).
#[test]
fn preprocess_image_is_deterministic() {
  let (w, h) = (200usize, 150usize);
  let base: Vec<f32> = (0..POS_EMBED_ELEMS)
    .map(|i| (i % 97) as f32 * 0.01)
    .collect();
  let rgb = synthetic_rgb(w, h);
  let a = preprocess_image(&rgb, w, h, &base, P).expect("a");
  let b = preprocess_image(&rgb, w, h, &base, P).expect("b");
  assert_eq!(a.grid, b.grid);
  assert_eq!(a.pixel_values, b.pixel_values);
  assert_eq!(a.attention_mask, b.attention_mask);
  assert_eq!(a.position_embeddings, b.position_embeddings);
}
