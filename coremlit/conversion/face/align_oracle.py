#!/usr/bin/env python3
"""Independent oracle for the ArcFace 5-point alignment golden.

Generates `coremlit/tests/face/fixtures/` — the committed expected pixels the
Rust `FaceAlign::to_template` golden is checked against.

WHAT IS INDEPENDENT HERE, AND WHAT IS A SPECIFICATION REPRODUCED TWICE
======================================================================
Stated precisely, because "oracle" is worth nothing if the reader has to guess
which half of it is evidence.

**The solve is independently derived.** InsightFace's `estimate_norm` calls
`skimage`'s `SimilarityTransform.estimate`, which is Umeyama (1991) by way of
an SVD with a determinant sign correction. Neither this file nor the Rust runs
an SVD: both reach the same minimiser through the complex/linear formulation,
where writing the scaled rotation as `[[a, -b], [b, a]]` makes the residual
linear in `(a, b, tx, ty)` and the answer a pair of dot products over the
centred point sets. The two agree to 1e-12 on the solved matrix. That agreement
is evidence about the DERIVATION, and it is also why the Rust's own tests do
not lean on it: `recovered_transform_is_the_least_squares_minimiser` proves
optimality by perturbation, naming no formula at all.

**The resampler is a specification reproduced twice, and is not independent
evidence about the algorithm.** `cv2.warpAffine`'s `INTER_LINEAR` is a
fixed-point pipeline, not a float bilinear kernel (see `warp_inter_linear`
below), and there is exactly one right answer to reproduce. What the golden
buys here is that two transcriptions of that pipeline — a scalar Rust loop and
the vectorised numpy below — land on the same 37 632 bytes, which catches a
transcription slip in either. It does not, and is not claimed to, establish
that the pipeline itself is OpenCV's. That claim rests on the constants being
named after the OpenCV symbols they come from, and on
`a_fraction_below_the_five_bit_half_step_takes_the_pure_left_pixel`, which
pins the one behaviour that separates the fixed-point pipeline from a float
one.

Reference semantics being reproduced (deepinsight/insightface, path
`python-package/insightface/utils/face_align.py`, pinned at commit
ffa12d315041c0505b077c7ff057ca914bb8dc7e, 2022-12-17):

    arcface_dst = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                   [41.5493, 92.3655], [70.7299, 92.2041]]
    tform.estimate(lmk, dst); M = tform.params[0:2, :]
    warped = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)

`cv2.warpAffine` without `WARP_INVERSE_MAP` inverts `M` itself and samples the
SOURCE at the inverse-mapped destination centre, `INTER_LINEAR`, constant-0
border. OpenCV **4.x** is the target: it is what the pinned `face_align.py`
runs against and what every published ArcFace accuracy number was measured on.
OpenCV 5.0 replaced this fixed-point path with a float one and is a different
function; see the Rust module doc.

Run: `python3 align_oracle.py` (numpy only — no skimage, no OpenCV, on purpose:
neither reference implementation is importable here, so nothing in this file
can silently become a call into the thing it is supposed to check).
"""

import hashlib
import pathlib

import numpy as np

# The ArcFace 112x112 destination template, in the landmark order the whole
# family uses: left eye, right eye, nose tip, left mouth corner, right mouth
# corner. Copied verbatim from the pinned `face_align.py` above.
ARCFACE_DST = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    # float32 THEN promoted, exactly as `face_align.py` declares it and as
    # `skimage` then promotes it. The Rust template is `f32` for the same
    # reason, so the two agree bit for bit rather than to within a rounding.
    dtype=np.float32,
).astype(np.float64)

TEMPLATE_SIZE = 112

# --- OpenCV's INTER_LINEAR fixed-point constants, by their own names ---------
INTER_BITS = 5                                    # imgproc/src/imgwarp.cpp
INTER_TAB_SIZE = 1 << INTER_BITS                  # 32
AB_BITS = max(10, INTER_BITS)                     # 10
AB_SCALE = 1 << AB_BITS                           # 1024
ROUND_DELTA = AB_SCALE // INTER_TAB_SIZE // 2     # 16
INTER_REMAP_COEF_BITS = 15
INTER_REMAP_COEF_SCALE = 1 << INTER_REMAP_COEF_BITS   # 32768

INT32_MIN = -(1 << 31)
INT32_MAX = (1 << 31) - 1


def cv_round(values):
    """OpenCV's `cvRound` / `saturate_cast<int>(double)`.

    `lrint` under the default rounding mode: nearest, TIES TO EVEN. The
    saturation stands in for C++'s undefined behaviour outside `int`.
    """
    return np.clip(np.rint(values), INT32_MIN, INT32_MAX).astype(np.int64)


def similarity_transform(src, dst):
    """Least-squares 2-D similarity `src -> dst`, complex/linear formulation.

    Minimises `sum ||S xi + t - yi||^2` over `S = [[a, -b], [b, a]]` and `t`.
    Centring removes `t`, and the residual is then linear in `(a, b)`:

        a = sum(Xi . Yi) / sum(|Xi|^2)      (dot)
        b = sum(Xi x Yi) / sum(|Xi|^2)      (cross, z component)

    Returns the 2x3 matrix `[S | t]`, the same object `tform.params[0:2, :]` is.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    assert src.shape == dst.shape and src.shape[1] == 2

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)
    x = src - src_mean
    y = dst - dst_mean

    denom = float((x * x).sum())
    if denom == 0.0:
        raise ValueError("degenerate landmarks: every point is the centroid")

    a = float((x[:, 0] * y[:, 0] + x[:, 1] * y[:, 1]).sum()) / denom
    b = float((x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0]).sum()) / denom

    s = np.array([[a, -b], [b, a]], dtype=np.float64)
    t = dst_mean - s @ src_mean
    return np.hstack([s, t.reshape(2, 1)])


def invert_affine(m):
    """`warpAffine`'s own inversion of `M`, in ITS operation order.

    Not `np.linalg.inv`: the resampler below is bit-exact, and a
    differently-associated inverse moves the sampled coordinate by an ulp and
    with it the occasional quantised pixel. OpenCV does, literally:

        D = M[0]*M[4] - M[1]*M[3];  D = D != 0 ? 1./D : 0;
        A11 = M[4]*D; A22 = M[0]*D;
        M[0] = A11; M[1] *= -D; M[3] *= -D; M[4] = A22;
        M[2] = -M[0]*M[2] - M[1]*M[5];
        M[5] = -M[3]*M[2] - M[4]*M[5];
    """
    m = [float(v) for v in np.asarray(m, dtype=np.float64).reshape(6)]
    d = m[0] * m[4] - m[1] * m[3]
    d = 1.0 / d if d != 0.0 else 0.0
    a11, a22 = m[4] * d, m[0] * d
    n0, n1, n3, n4 = a11, m[1] * -d, m[3] * -d, a22
    n2 = -n0 * m[2] - n1 * m[5]
    n5 = -n3 * m[2] - n4 * m[5]
    return [n0, n1, n2, n3, n4, n5]


def warp_inter_linear(img, m, size):
    """`cv2.warpAffine(img, m, (size, size), INTER_LINEAR, BORDER_CONSTANT, 0)`.

    `m` maps source -> destination, so the sampler uses its inverse.

    This is NOT a float bilinear kernel with the weights rounded at the end.
    OpenCV quantises the inverse-mapped coordinate onto a five-bit grid before
    it picks a weight at all, so a true fraction below the half-step 1/64
    collapses to zero and the tap is the pure left pixel. Reproduced here in
    the order `imgwarp.cpp`'s `WarpAffineInvoker` does it, because the
    intermediate roundings are part of the answer:

      * the per-destination-COLUMN contribution is rounded to 1/AB_SCALE of a
        pixel on its own (`adelta`/`bdelta`), and so is the per-ROW half;
      * only then are the two added and shifted onto the 1/INTER_TAB_SIZE grid,
        with `ROUND_DELTA` folded in to make that truncation a rounding;
      * the four interpolation weights are 15-bit integers summing to exactly
        `1 << 15` (`BilinearTab_i`), the taps accumulate as integers, and the
        result is `(acc + (1 << 14)) >> 15` saturated into uint8.

    `cvRound(a*u*1024) + cvRound(m*v*1024)` is not `cvRound((a*u + m*v)*1024)`,
    which is exactly why the two halves are rounded separately here.
    """
    inv = invert_affine(m)
    height, width, channels = img.shape
    src = img.astype(np.int64)

    axis = np.arange(size, dtype=np.float64)
    adelta = cv_round(inv[0] * axis * AB_SCALE)          # per destination column
    bdelta = cv_round(inv[3] * axis * AB_SCALE)
    x0 = cv_round((inv[1] * axis + inv[2]) * AB_SCALE) + ROUND_DELTA   # per row
    y0 = cv_round((inv[4] * axis + inv[5]) * AB_SCALE) + ROUND_DELTA

    # `>>` is arithmetic on numpy's signed integers, so it floors for negative
    # coordinates exactly as C++'s does.
    x = (x0[:, None] + adelta[None, :]) >> (AB_BITS - INTER_BITS)
    y = (y0[:, None] + bdelta[None, :]) >> (AB_BITS - INTER_BITS)

    # `saturate_cast<short>` on the integer tap; the low INTER_BITS are the
    # fraction's index into the weight table.
    sx = np.clip(x >> INTER_BITS, -32768, 32767)
    sy = np.clip(y >> INTER_BITS, -32768, 32767)
    fx = x & (INTER_TAB_SIZE - 1)
    fy = y & (INTER_TAB_SIZE - 1)

    # BilinearTab_i[fy*INTER_TAB_SIZE + fx], as exact integers. OpenCV's own
    # table differs in one of its 1024 cells: `initInterTab2D` builds it with
    # `saturate_cast<short>`, so the unit weight at fraction (0, 0) saturates
    # to 32767 and its sum-fixing step moves the missing 1 to the opposite
    # corner. For a uint8 source the two tables are the same function (the Rust
    # gate `the_saturating_weight_table_cell_is_invisible_for_u8_sources`
    # proves it exhaustively), so the exact form is used here.
    weights = [
        (INTER_TAB_SIZE - fy) * (INTER_TAB_SIZE - fx) * INTER_TAB_SIZE,
        (INTER_TAB_SIZE - fy) * fx * INTER_TAB_SIZE,
        fy * (INTER_TAB_SIZE - fx) * INTER_TAB_SIZE,
        fy * fx * INTER_TAB_SIZE,
    ]

    acc = np.zeros((size, size, channels), dtype=np.int64)
    for (dy, dx), weight in zip(((0, 0), (0, 1), (1, 0), (1, 1)), weights):
        xx, yy = sx + dx, sy + dy
        # BORDER_CONSTANT with borderValue 0: an out-of-range tap contributes
        # `0 * weight`. The bounds are tested on the QUANTISED tap, as OpenCV
        # tests them.
        inside = (xx >= 0) & (xx < width) & (yy >= 0) & (yy < height)
        taps = src[np.clip(yy, 0, height - 1), np.clip(xx, 0, width - 1), :]
        acc += (weight * inside)[:, :, None] * taps

    # FixedPtCast<int, uchar, INTER_REMAP_COEF_BITS>.
    half = 1 << (INTER_REMAP_COEF_BITS - 1)
    return np.clip((acc + half) >> INTER_REMAP_COEF_BITS, 0, 255).astype(np.uint8)


def synthetic_crop(width, height):
    """The committed source crop.

    Deliberately NOT square (a width/height transposition changes it), with a
    different generator per channel so a channel permutation changes it too,
    and with the modulo wrap-around kept because a discontinuity is what makes
    a wrong interpolation weight visible instead of merely inaccurate.
    """
    xs = np.arange(width, dtype=np.int64)[None, :]
    ys = np.arange(height, dtype=np.int64)[:, None]
    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[:, :, 0] = ((17 * xs + 3 * ys) % 256).astype(np.uint8)
    img[:, :, 1] = ((5 * xs + 29 * ys) % 256).astype(np.uint8)
    img[:, :, 2] = ((xs * ys) % 251).astype(np.uint8)
    return img


# Literal landmarks. NOT derived from `ARCFACE_DST`: a golden whose landmarks
# are a similarity image of the template moves WITH the template, so a mutated
# template coordinate would leave the expected pixels unchanged and the golden
# would pass over the mutation. These are fixed numbers, so the template's own
# values are load-bearing for the committed bytes.
CROP_WIDTH = 64
CROP_HEIGHT = 48
LANDMARKS = np.array(
    [
        [18.5, 16.0],
        [41.0, 13.5],
        [30.5, 25.0],
        [21.0, 35.5],
        [40.0, 33.0],
    ],
    dtype=np.float64,
)


def main():
    here = pathlib.Path(__file__).resolve().parent
    out_dir = here.parent.parent / "tests" / "face" / "fixtures"
    out_dir.mkdir(parents=True, exist_ok=True)

    crop = synthetic_crop(CROP_WIDTH, CROP_HEIGHT)
    m = similarity_transform(LANDMARKS, ARCFACE_DST)
    aligned = warp_inter_linear(crop, m, TEMPLATE_SIZE)

    crop_path = out_dir / "align_crop_64x48_rgb8.bin"
    aligned_path = out_dir / "align_expected_112x112_rgb8.bin"
    crop_path.write_bytes(crop.tobytes())
    aligned_path.write_bytes(aligned.tobytes())

    print("transform (source -> template), row-major 2x3:")
    for row in m:
        print("   ", ", ".join(repr(float(v)) for v in row))
    print()
    for path in (crop_path, aligned_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.name}: {path.stat().st_size} bytes, sha256 {digest}")


if __name__ == "__main__":
    main()
