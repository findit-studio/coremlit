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

**The solve is `f64` where skimage's is `f32`, and that is a divergence rather
than a rounding.** It moves five-bit source coordinates. It also cannot be
closed, because skimage's `f32` path delegates a `sgemm` and a `sgesdd` and two
correct BLAS builds disagree by more than this crate disagrees with either.
`--reference-divergence` measures all of it; the `align` module doc and this
directory's README carry the conclusion.

Run: `python3 align_oracle.py` to regenerate the fixtures (numpy only — no
skimage, no OpenCV, on purpose: neither reference implementation is importable
here, so nothing in this file can silently become a call into the thing it is
supposed to check), or `python3 align_oracle.py --reference-divergence` to
reprint the reference matrices and divergence counts `align/tests.rs` commits.
"""

import hashlib
import pathlib
import sys

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


def five_bit_source_grid(m, size=TEMPLATE_SIZE):
    """The destination -> five-bit source coordinate map `warpAffine` walks.

    `m` maps source -> destination; the sampler uses its inverse. Split out of
    `warp_inter_linear` because the intermediate roundings are part of the
    answer AND because two matrices are compared on the coordinates rather
    than on pixels: a moved coordinate leaves the output unchanged wherever it
    lands in a flat neighbourhood, so differing bytes under-report a moved map.
    The Rust `SourceGrid` is the same split.
    """
    inv = invert_affine(m)
    axis = np.arange(size, dtype=np.float64)
    adelta = cv_round(inv[0] * axis * AB_SCALE)          # per destination column
    bdelta = cv_round(inv[3] * axis * AB_SCALE)
    x0 = cv_round((inv[1] * axis + inv[2]) * AB_SCALE) + ROUND_DELTA   # per row
    y0 = cv_round((inv[4] * axis + inv[5]) * AB_SCALE) + ROUND_DELTA
    # `>>` is arithmetic on numpy's signed integers, so it floors for negative
    # coordinates exactly as C++'s does.
    return ((x0[:, None] + adelta[None, :]) >> (AB_BITS - INTER_BITS),
            (y0[:, None] + bdelta[None, :]) >> (AB_BITS - INTER_BITS))


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
    x, y = five_bit_source_grid(m, size)
    height, width, channels = img.shape
    src = img.astype(np.int64)

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


# --- `--reference-divergence`: how far the solve is from skimage, and how far
# --- skimage is from itself ------------------------------------------------
#
# skimage's `_umeyama` keeps its float32 input through the centroids, the
# covariance and the SVD (`skimage/transform/_geometric.py` v0.19.3, L107-149);
# the Rust and the golden above both promote to float64 first. Same minimiser,
# different numbers. This section measures the difference AND the reason it
# cannot be closed: `_umeyama`'s float32 path is two library calls, a `sgemm`
# and a `sgesdd`, and neither is specified beyond returning *a* correct answer.
#
# Still no skimage import. The functions below are transcribed from the source
# cited above and verified against the numpy expression they stand for; what
# varies is only which BLAS/LAPACK performs the two calls skimage delegates.

WITNESS = np.array(
    [
        [48.073643, 97.0597],
        [103.45303, 115.63326],
        [68.99921, 127.54772],
        [37.211536, 152.98666],
        [82.01403, 169.19621],
    ],
    dtype=np.float32,
)

ARCFACE_DST_F32 = ARCFACE_DST.astype(np.float32)


def f32_mean_axis0(a):
    """numpy's `a.mean(axis=0)` for float32: sequential sum over the outer
    axis, then a float32 divide."""
    acc = a[0].copy()
    for i in range(1, a.shape[0]):
        acc = np.float32(acc + a[i])
    return np.float32(acc / np.float32(a.shape[0]))


def f32_var_axis0(a):
    """numpy's two-pass `a.var(axis=0)`: re-centre on the float32 mean, square,
    sum, divide -- all in float32. NOT `mean(a**2)`: `a` is already demeaned in
    float32, so its float32 mean is near zero but not zero, and re-centring on
    it is part of the answer."""
    x = np.float32(a - f32_mean_axis0(a))
    x = np.float32(x * x)
    acc = x[0].copy()
    for i in range(1, x.shape[0]):
        acc = np.float32(acc + x[i])
    return np.float32(acc / np.float32(x.shape[0]))


def umeyama_f32(src, dst, gemm, svd):
    """`_umeyama(src, dst, estimate_scale=True)`, operand dtypes exactly as the
    source has them: `src`/`dst` stay float32, `d` and `T` are float64.

    `gemm` and `svd` are the two calls skimage delegates to numpy and numpy
    delegates to whatever BLAS/LAPACK it was linked against. Passing them in is
    the whole point: the same source, the same inputs, two correct libraries.
    """
    num, dim = src.shape
    src_mean = f32_mean_axis0(src)
    dst_mean = f32_mean_axis0(dst)
    src_demean = np.float32(src - src_mean)
    dst_demean = np.float32(dst - dst_mean)
    A = np.float32(gemm(np.ascontiguousarray(dst_demean.T), src_demean) / np.float32(num))
    d = np.ones((dim,), dtype=np.double)
    if np.linalg.det(A.astype(np.float64)) < 0:
        d[dim - 1] = -1
    U, S, V = svd(A)
    R = (U.astype(np.float64) @ np.diag(d)) @ V.astype(np.float64)
    var = f32_var_axis0(src_demean)
    var_sum = np.float32(var[0] + var[1])
    scale = float(np.float32(np.float64(1.0) / var_sum)) * float(S.astype(np.float64) @ d)
    t = dst_mean.astype(np.float64) - scale * (R @ src_mean.astype(np.float64))
    return np.hstack([R * scale, t.reshape(2, 1)])


def numpy_backend():
    """Whatever BLAS/LAPACK this numpy was built against."""
    return (lambda a, b: (a @ b).astype(np.float32)), np.linalg.svd


def accelerate_backend():
    """Apple's Accelerate, by ctypes -- a second, independent BLAS/LAPACK on
    the same machine, so "bit-exact with skimage" can be tested for being a
    property of skimage at all. `None` where it is not available.

    This is not a call into the thing being checked: the Rust runs no LAPACK,
    and the reference implementation (skimage) is still not imported. It is a
    second performer of the two calls skimage itself delegates.
    """
    import ctypes

    try:
        acc = ctypes.CDLL("/System/Library/Frameworks/Accelerate.framework/Accelerate")
    except OSError:
        return None
    ptr = lambda a: a.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    acc.cblas_sgemm.restype = None
    acc.cblas_sgemm.argtypes = [ctypes.c_int] * 6 + [
        ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_float,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
    ]
    acc.sgesdd_.restype = None

    def gemm(a, b):
        a = np.ascontiguousarray(a, np.float32)
        b = np.ascontiguousarray(b, np.float32)
        m, k = a.shape
        n = b.shape[1]
        c = np.zeros((m, n), np.float32)
        acc.cblas_sgemm(101, 111, 111, m, n, k, ctypes.c_float(1.0), ptr(a), k,
                        ptr(b), n, ctypes.c_float(0.0), ptr(c), n)
        return c

    def svd(a2x2):
        n = 2
        a = np.asfortranarray(a2x2, np.float32).copy(order="F")
        s = np.zeros(n, np.float32)
        u = np.zeros((n, n), np.float32, order="F")
        vt = np.zeros((n, n), np.float32, order="F")
        work = np.zeros(512, np.float32)
        iwork = np.zeros(8 * n, np.int32)
        info = ctypes.c_int(0)
        ci = lambda v: ctypes.byref(ctypes.c_int(v))
        acc.sgesdd_(ctypes.c_char_p(b"A"), ci(n), ci(n), ptr(a), ci(n), ptr(s),
                    ptr(u), ci(n), ptr(vt), ci(n), ptr(work), ci(512),
                    iwork.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                    ctypes.byref(info))
        if info.value != 0:
            raise RuntimeError(f"sgesdd_ returned {info.value}")
        return np.array(u), np.array(s), np.array(vt)

    # The bridge asserts itself before anything is measured through it: a
    # comparison against a mis-wired ctypes call would report a divergence that
    # is entirely this function's.
    rng = np.random.default_rng(3)
    for _ in range(50):
        a = (rng.standard_normal((2, 5)) * 50).astype(np.float32)
        b = (rng.standard_normal((5, 2)) * 50).astype(np.float32)
        if not np.allclose(gemm(a, b).astype(float),
                           a.astype(float) @ b.astype(float), rtol=1e-4, atol=1e-1):
            raise RuntimeError("Accelerate sgemm bridge does not reproduce a matmul")
        m = (rng.standard_normal((2, 2)) * 300).astype(np.float32)
        u, s, vt = svd(m)
        if not np.allclose(u @ np.diag(s) @ vt, m, atol=2e-2):
            raise RuntimeError("Accelerate sgesdd bridge does not reconstruct its input")
    return gemm, svd


def coordinate_divergence(left, right):
    """Destination pixels whose five-bit source coordinate differs."""
    xl, yl = five_bit_source_grid(np.asarray(left).reshape(2, 3))
    xr, yr = five_bit_source_grid(np.asarray(right).reshape(2, 3))
    return int(((xl != xr) | (yl != yr)).sum())


def face_like_landmarks(n, seed=31337):
    """Landmark sets shaped like real detections: a similarity image of the
    template plus noise, so `det(A) > 0` the way an alignment's always is."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        s = rng.uniform(0.5, 4.0)
        th = rng.uniform(-0.6, 0.6)
        rot = np.array([[s * np.cos(th), -s * np.sin(th)],
                        [s * np.sin(th), s * np.cos(th)]])
        t = rng.uniform(-50, 250, size=2)
        p = (ARCFACE_DST @ rot.T + t) + rng.normal(0, 1.5, size=(5, 2))
        out.append(p.astype(np.float32))
    return out


def sweep(n, backends):
    """The bulk statistics the module doc and README quote.

    Every one is a comparison between two CORRECT libraries on identical
    inputs, so a difference is evidence that the reference is under-specified
    rather than that one of them is wrong.
    """
    print(f"\n--- {n} face-like landmark sets ---")
    dst_demean = np.float32(ARCFACE_DST_F32 - f32_mean_axis0(ARCFACE_DST_F32))
    sets = face_like_landmarks(n)

    if len(backends) == 2:
        (n1, (g1, s1)), (n2, (g2, s2)) = backends
        cov = sv = uv = 0
        for src in sets:
            src_demean = np.float32(src - f32_mean_axis0(src))
            a1 = g1(np.ascontiguousarray(dst_demean.T), src_demean)
            a2 = g2(np.ascontiguousarray(dst_demean.T), src_demean)
            if not np.array_equal(a1, a2):
                cov += 1
            a = np.float32(a1 / np.float32(5))       # the SAME A into both SVDs
            u1, sing1, v1 = s1(a)
            u2, sing2, v2 = s2(a)
            if not np.array_equal(sing1, sing2):
                sv += 1
            if not np.array_equal(u1.astype(np.float64) @ v1.astype(np.float64),
                                  u2.astype(np.float64) @ v2.astype(np.float64)):
                uv += 1
        print(f"{n1} vs {n2}:")
        print(f"    f32 covariance sgemm differs      : {cov} of {n}")
        print(f"    f32 sgesdd singular values differ : {sv} of {n}")
        print(f"    f32 sgesdd rotation U@V differs   : {uv} of {n}")

    for name, (gemm, svd) in backends:
        not_sim = sum(
            not (m[0, 0] == m[1, 1] and m[0, 1] == -m[1, 0])
            for m in (umeyama_f32(s, ARCFACE_DST_F32, gemm, svd) for s in sets)
        )
        print(f"    _umeyama results under {name} that are NOT a similarity: "
              f"{not_sim} of {n}")

    if len(backends) == 2:
        counts = [
            coordinate_divergence(umeyama_f32(s, ARCFACE_DST_F32, *backends[0][1]),
                                  umeyama_f32(s, ARCFACE_DST_F32, *backends[1][1]))
            for s in sets
        ]
        print(f"    five-bit coordinates, {backends[0][0]} vs {backends[1][0]}: "
              f"mean {np.mean(counts):.1f}, median {np.median(counts):.0f}, "
              f"max {max(counts)}")


def identify_gemm_accumulation(trials=3000, seed=11):
    """WHICH accumulation numpy's f32 (2x5)@(5x2) performs.

    The mechanism behind the covariance divergence: OpenBLAS's aarch64 sgemm
    contracts its multiply-adds into `fma`, and whether a kernel contracts is a
    build flag rather than a specification.
    """
    import math
    rng = np.random.default_rng(seed)

    def seq(x, y):
        acc = np.float32(0.0)
        for i in range(len(x)):
            acc = np.float32(acc + np.float32(x[i] * y[i]))
        return acc

    def fma(x, y):
        acc = 0.0
        for i in range(len(x)):
            acc = float(np.float32(math.fma(float(x[i]), float(y[i]), acc)))
        return np.float32(acc)

    hits = {"sequential (unfused)": 0, "fma chain": 0}
    for _ in range(trials):
        a = (rng.standard_normal((2, 5)) * 60).astype(np.float32)
        b = (rng.standard_normal((5, 2)) * 60).astype(np.float32)
        ref = a @ b
        for label, fn in (("sequential (unfused)", seq), ("fma chain", fma)):
            got = np.array([[fn(a[i], b[:, j]) for j in range(2)] for i in range(2)],
                           dtype=np.float32)
            if np.array_equal(ref, got):
                hits[label] += 1
    print(f"\n--- what this numpy's f32 (2x5)@(5x2) actually accumulates "
          f"({trials} random pairs) ---")
    for label, count in hits.items():
        print(f"    {label:<22}: reproduces {count} of {trials}")


def reference_divergence(sweep_size=0):
    """Print the constants and counts `align/tests.rs` commits."""
    solved = similarity_transform(WITNESS.astype(np.float64), ARCFACE_DST)
    backends = [("numpy/OpenBLAS", numpy_backend())]
    accel = accelerate_backend()
    if accel is not None:
        backends.append(("Apple Accelerate", accel))
    else:
        print("NOTE: Accelerate unavailable; the two-build spread cannot be "
              "measured here and the committed Accelerate matrix stands.\n")

    print("witness landmarks (float32, as a detector emits them):")
    for p in WITNESS:
        print(f"    [{float(p[0])!r}, {float(p[1])!r}]")
    print("\nsolve (this crate: exact f64 minimiser of the f32 landmarks):")
    for v in np.asarray(solved).reshape(6):
        print(f"    {float(v)!r},")

    results = {}
    for name, (gemm, svd) in backends:
        m = umeyama_f32(WITNESS, ARCFACE_DST_F32, gemm, svd)
        results[name] = m
        sim = (m[0, 0] == m[1, 1]) and (m[0, 1] == -m[1, 0])
        print(f"\nskimage f32 _umeyama under {name}"
              f"{'' if sim else '   (NOT a similarity: a != d or b != -c)'}:")
        for v in np.asarray(m).reshape(6):
            print(f"    {float(v)!r},")

    print("\ndestination pixels (of 12544) whose five-bit source coordinate differs:")
    for name, m in results.items():
        print(f"    solve            vs {name:<18}: {coordinate_divergence(solved, m)}")
    names = list(results)
    if len(names) == 2:
        print(f"    {names[0]:<16} vs {names[1]:<18}: "
              f"{coordinate_divergence(results[names[0]], results[names[1]])}"
              "   <- the reference against ITSELF")

    if sweep_size:
        identify_gemm_accumulation()
        sweep(sweep_size, backends)


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
    if "--reference-divergence" in sys.argv:
        size = 0
        if "--sweep" in sys.argv:
            size = int(sys.argv[sys.argv.index("--sweep") + 1])
        reference_divergence(size)
    else:
        main()
