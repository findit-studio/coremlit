# `embeddings::face` — the alignment oracle

This directory holds no model conversion, because **`face` stages no model
artifact**. See `coremlit/src/embeddings/face/mod.rs`'s module doc and
`coremlit/FEATURE_MAP.md` for why that is a finding about the licence policy
rather than an omission, and what has to change before a conversion recipe
belongs here.

What it does hold is the **oracle that produces the alignment golden** — the
committed expected pixels `tests/face/align_golden.rs` compares against.
Alignment is the one place in this kit where a wrong answer raises no error and
moves every downstream cosine, so it gets an oracle written independently of
the code it checks.

## `align_oracle.py`

Reproduces `deepinsight/insightface`'s `estimate_norm` + `norm_crop`
(`python-package/insightface/utils/face_align.py`, commit
`ffa12d315041c0505b077c7ff057ca914bb8dc7e`, 2022-12-17), and writes two raw
RGB8 fixtures into `coremlit/tests/face/fixtures/`.

It imports **numpy only**. `skimage` and OpenCV are deliberately not used (and
are not installed here), so nothing in this file can quietly become a call into
the thing it is meant to check.

**What is independent, and what is a specification reproduced twice** — stated
here because "oracle" is worth nothing if the reader has to guess which half is
evidence:

- the **solve** is independently derived. InsightFace's `estimate_norm` calls
  `skimage`'s `SimilarityTransform.estimate`, which is Umeyama (1991) by way of
  an SVD with a determinant sign correction; neither this script nor the Rust
  runs an SVD, both reaching the same minimiser through the **complex/linear**
  formulation, where writing the scaled rotation as `[[a, -b], [b, a]]` makes
  the residual linear in `(a, b, tx, ty)` and the answer two dot products. The
  Rust's own optimality gate (`recovered_transform_is_the_least_squares_minimiser`)
  names no formula at all, so the golden is not the only leg. **Both sides
  evaluate that minimiser in `f64` where `skimage` evaluates it in `f32`, which
  is a divergence and not a rounding** — see "The solve is not bit-exact with
  `skimage`" below;
- the **resampler** is not independent. `cv2.warpAffine`'s `INTER_LINEAR` has
  exactly one right answer and both sides reproduce it, so byte agreement here
  catches a transcription slip and nothing more.

**The resampler is bit-exact with OpenCV, and that replaced a false claim.**
This file and the Rust module used to resample in `f64` and record the
divergence from OpenCV as "at most one LSB per channel". That was a measurement
on one fixture stated as a bound over the domain, and it is false: OpenCV
quantises the inverse-mapped coordinate onto a five-bit grid *before* choosing
a weight, so a fraction under `1/64` collapses to zero and takes the pure left
pixel. Measured against `cv2.warpAffine` over ArcFace-shaped warps of random
crops (`opencv-python-headless` 4.12.0, 451 584 bytes), the float sampler
differed on **11.6 % of bytes, worst case 6 levels**. Both files now implement
OpenCV's fixed-point pipeline, with every constant named after the OpenCV
symbol it comes from.

**Which OpenCV — the version is part of the contract.** The 4.x line: it is
what the pinned `face_align.py` runs against and what every published ArcFace
accuracy number was measured on. OpenCV **5.0 replaced the fixed-point path
with a float one** and is a different function — against 5.0.0 these same
committed bytes differ on 8 488 of 37 632, by up to 5 levels.

## The solve is not bit-exact with `skimage`, and there is no single `skimage` to be exact against

`skimage`'s `_umeyama` (`skimage/transform/_geometric.py` v0.19.3, L107-149)
keeps its **`f32`** input through the centroids, the covariance and the SVD,
storing only the result as `f64`. This script and the Rust promote to `f64`
first. Same minimiser, different numbers — enough to move a five-bit source
coordinate on **10 of 12 544** destination pixels for the witness landmarks
`align_oracle.py --reference-divergence` reports.

That gap is real. What makes it unclosable is that `_umeyama`'s `f32` path is
two library calls — a `sgemm` for the covariance, a `sgesdd` for the 2×2 SVD —
and neither is specified past returning *a* correct answer. Running the
identical `_umeyama` source, same machine, same landmarks, under numpy's
OpenBLAS 0.3.33 and Apple's Accelerate:

| measured over | OpenBLAS vs Accelerate |
|---|---|
| `f32` covariance, face-like landmark sets | 16 618 of 20 000 differ |
| `f32` `sgesdd` singular values | 13 657 of 20 000 differ |
| `f32` `sgesdd` rotation `U @ V` | 20 000 of 20 000 differ |
| `_umeyama` results that are not a similarity | 19 624 of 20 000 (Accelerate); 0 (OpenBLAS) |
| five-bit coordinates, the witness | **15** of 12 544 differ |
| five-bit coordinates, 20 000 face-like sets | mean **14.8**, median 11, worst 212 |

The two builds disagree with **each other** by more than this crate disagrees
with either (10 against OpenBLAS, 5 against Accelerate). OpenBLAS's aarch64
`sgemm` contracts its multiply-adds into `fma` — an `fma` chain reproduces it on
3 000 of 3 000 random inputs, a non-fused chain on 341 — and whether a kernel
contracts is a build flag, not a specification.

There is a structural obstruction too: a `f32` `U @ V` is only approximately
orthogonal, so under Accelerate 98 % of `_umeyama` results are not exactly
similarities — and under OpenBLAS none of them fail to be, which is itself the
point: the property belongs to the build, not to the reference. Rust's
`SimilarityTransform` stores `(a, b)` and cannot represent a shear, so the
reference's own output is routinely not a value that type can hold.

So "bit-exact with `skimage`" is a property of `skimage` *and the BLAS the
measuring machine linked*, not of `skimage`. This crate therefore claims only
what it can hold: the least-squares similarity minimiser of the `f32`
landmarks, evaluated in `f64` rather than in the reference's `f32`, handed to a
resampler that is bit-exact with `cv2.warpAffine` 4.x.
`the_solve_diverges_from_skimage_by_less_than_skimage_diverges_from_itself`
commits all three counts, so closing the gap toward one build turns that test
red on the gap that has no single target.

Deciding it on accuracy rather than on bit-exactness needs the embedding drift
the divergence causes, measured against a staged artifact. There is none (see
`src/embeddings/face/mod.rs`), so the divergence is recorded rather than traded
away.

```sh
python3 coremlit/conversion/face/align_oracle.py --reference-divergence --sweep 20000
```

prints every number on this page: the witness, all three matrices, the three
witness counts that `src/embeddings/face/align/tests.rs` commits as constants,
the accumulation identification, and the table above (about ten seconds; drop
`--sweep` for the matrices and the three counts alone). **This mode, unlike the
golden,
depends on the BLAS**: that is what it exists to demonstrate. It needs no
skimage and no OpenCV, only numpy plus — for the second backend — Apple's
Accelerate through `ctypes`, whose bridge asserts that it reproduces a matmul
and reconstructs an SVD before anything is measured through it. Where
Accelerate is unavailable the mode says so and reports the one backend it has.

## Regenerating

```sh
python3 coremlit/conversion/face/align_oracle.py
```

It prints the solved 2×3 matrix and both fixtures' SHA-256. Those digests are
pinned in `tests/face/align_golden.rs` (`CROP_SHA256`, `EXPECTED_SHA256`) and
the matrix in `ORACLE_MATRIX`, so a regeneration is a deliberate three-place
diff and never a silent re-baseline.

## Observed toolchain (per #97: observed, not a literal)

The committed fixtures were produced by this exact stack:

| component | observed |
|---|---|
| macOS | 26.5, arm64 |
| python | 3.14.6 (`/opt/homebrew/opt/python@3.14/bin/python3.14`) |
| numpy | 2.5.1 |
| `align_oracle.py` sha256 | `e96bf45ab2a5a712e7b5f2f9029958cced273f8172f36a483ba0594dede30ba4` |
| `align_crop_64x48_rgb8.bin` sha256 | `a7d34a19107058c28c73633cc25b82a018fc279034d6670b45488022d5071ce0` |
| `align_expected_112x112_rgb8.bin` sha256 | `274b92b0002ab01af0c8967372b2aea7bf5a71096308f257ea50168cf671f13c` |

The solve is IEEE-754 `f64` throughout with no BLAS call on the hot path and
the resampler is integer arithmetic after it, so a different numpy or Python is
expected to reproduce these **fixture** bytes; the row records what was actually
run, not what is required.

`--reference-divergence` is the exception and deliberately so: its numbers are
BLAS-dependent by construction, and the row above pins the build the committed
`SKIMAGE_OPENBLAS` constant came from (numpy 2.5.1, OpenBLAS 0.3.33, aarch64).
A different build will print a different matrix, which is the finding.

The OpenCV comparisons quoted above were run **out of tree**, in a throwaway
virtualenv, purely to check this reproduction against the thing it reproduces.
Nothing in the repository imports OpenCV and no gate depends on it: the numbers
are recorded here as an observation, and what the committed tests actually hold
is the Rust and this script agreeing on all 37 632 bytes plus the unit gate
`a_fraction_below_the_five_bit_half_step_takes_the_pure_left_pixel`, which pins
the one behaviour separating the fixed-point pipeline from a float one.
