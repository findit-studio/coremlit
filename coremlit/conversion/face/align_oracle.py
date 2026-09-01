#!/usr/bin/env python3
"""Independent oracle for the ArcFace 5-point alignment golden.

Generates `coremlit/tests/face/fixtures/` — the committed expected pixels the
Rust `FaceAlign::to_template` golden is checked against.

WHY THIS IS AN ORACLE AND NOT A SECOND COPY OF THE IMPLEMENTATION
=================================================================
The Rust implementation solves the 5-point similarity transform with the
**Umeyama (1991) SVD** construction, which is what `skimage`'s
`SimilarityTransform.estimate` — the function InsightFace's `estimate_norm`
calls — uses.

This script deliberately does NOT. It solves the same least-squares problem
through the **complex/linear formulation**: writing the scaled rotation as
`[[a, -b], [b, a]]` makes the residual linear in `(a, b, tx, ty)`, so the
minimiser is a pair of dot products over the centred point sets and needs no
SVD, no determinant sign correction, and no eigen decomposition. Two different
derivations of one minimiser agreeing to float precision is evidence; one
derivation run twice is not.

Reference semantics being reproduced (deepinsight/insightface, path
`python-package/insightface/utils/face_align.py`, pinned at commit
ffa12d315041c0505b077c7ff057ca914bb8dc7e, 2022-12-17):

    arcface_dst = [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                   [41.5493, 92.3655], [70.7299, 92.2041]]
    tform.estimate(lmk, dst); M = tform.params[0:2, :]
    warped = cv2.warpAffine(img, M, (112, 112), borderValue=0.0)

`cv2.warpAffine` without `WARP_INVERSE_MAP` inverts `M` itself and samples the
SOURCE at the inverse-mapped destination centre, `INTER_LINEAR`, constant-0
border. That is what `warp_bilinear` below does — in float64, where OpenCV uses
5-bit fixed-point interpolation weights for 8-bit images. The two differ by at
most one LSB per channel; see the Rust golden's module doc for why that is far
below the ANE's own fp16 floor and is recorded rather than chased.

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


def warp_bilinear(img, m, size):
    """`cv2.warpAffine(img, m, (size, size), borderValue=0)` semantics.

    `m` maps source -> destination, so the sampler uses its inverse. Bilinear,
    pixel centres at integer coordinates, out-of-range taps contribute 0, and
    the result is rounded half-away-from-zero into uint8.
    """
    inv = np.linalg.inv(np.vstack([m, [0.0, 0.0, 1.0]]))[0:2, :]
    height, width, channels = img.shape
    src = img.astype(np.float64)
    out = np.zeros((size, size, channels), dtype=np.float64)

    for v in range(size):
        for u in range(size):
            fx = inv[0, 0] * u + inv[0, 1] * v + inv[0, 2]
            fy = inv[1, 0] * u + inv[1, 1] * v + inv[1, 2]
            x0 = int(np.floor(fx))
            y0 = int(np.floor(fy))
            ax = fx - x0
            ay = fy - y0
            acc = np.zeros(channels, dtype=np.float64)
            for dy, wy in ((0, 1.0 - ay), (1, ay)):
                for dx, wx in ((0, 1.0 - ax), (1, ax)):
                    w = wy * wx
                    if w == 0.0:
                        continue
                    xx = x0 + dx
                    yy = y0 + dy
                    if 0 <= xx < width and 0 <= yy < height:
                        acc += w * src[yy, xx, :]
            out[v, u, :] = acc

    return np.clip(np.floor(out + 0.5), 0.0, 255.0).astype(np.uint8)


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
    aligned = warp_bilinear(crop, m, TEMPLATE_SIZE)

    crop_path = out_dir / "align_crop_64x48_rgb8.bin"
    aligned_path = out_dir / "align_expected_112x112_rgb8.bin"
    crop_path.write_bytes(crop.tobytes())
    aligned_path.write_bytes(aligned.tobytes())

    print("transform (source -> template), row-major 2x3:")
    for row in m:
        print("   ", ", ".join(f"{v!r}" for v in row))
    print()
    for path in (crop_path, aligned_path):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(f"{path.name}: {path.stat().st_size} bytes, sha256 {digest}")


if __name__ == "__main__":
    main()
