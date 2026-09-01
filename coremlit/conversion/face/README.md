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
  names no formula at all, so the golden is not the only leg;
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
| `align_oracle.py` sha256 | `d28a94294dc8f82783771b8026a117e3550d227762a3bd640b7ad27454947b53` |
| `align_crop_64x48_rgb8.bin` sha256 | `a7d34a19107058c28c73633cc25b82a018fc279034d6670b45488022d5071ce0` |
| `align_expected_112x112_rgb8.bin` sha256 | `274b92b0002ab01af0c8967372b2aea7bf5a71096308f257ea50168cf671f13c` |

The solve is IEEE-754 `f64` throughout with no BLAS call on the hot path and
the resampler is integer arithmetic after it, so a different numpy or Python is
expected to reproduce these bytes; the row records what was actually run, not
what is required.

The OpenCV comparisons quoted above were run **out of tree**, in a throwaway
virtualenv, purely to check this reproduction against the thing it reproduces.
Nothing in the repository imports OpenCV and no gate depends on it: the numbers
are recorded here as an observation, and what the committed tests actually hold
is the Rust and this script agreeing on all 37 632 bytes plus the unit gate
`a_fraction_below_the_five_bit_half_step_takes_the_pure_left_pixel`, which pins
the one behaviour separating the fixed-point pipeline from a float one.
