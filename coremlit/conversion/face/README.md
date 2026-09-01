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

It is an oracle rather than a second copy of the implementation:

- the Rust follows **Umeyama's** statement of the least-squares similarity;
  this script solves the same minimiser through the **complex/linear**
  formulation, where writing the scaled rotation as `[[a, -b], [b, a]]` makes
  the residual linear in `(a, b, tx, ty)` and the answer two dot products — no
  SVD, no determinant sign correction;
- it imports **numpy only**. `skimage` and OpenCV are deliberately not used
  (and are not installed here), so nothing in this file can quietly become a
  call into the thing it is meant to check.

Two divergences from OpenCV are deliberate and are recorded in the Rust
module's doc: `f64` bilinear weights rather than OpenCV's 5-bit fixed point,
and half-up rounding rather than half-to-even. Both are bounded by one LSB per
channel, far below the ANE's own fp16 embedding floor.

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
| `align_oracle.py` sha256 | `16cb02817146c095cd2bf5eb3ff0e1794dca296a58129fc7fd482c855eb4d5d6` |
| `align_crop_64x48_rgb8.bin` sha256 | `a7d34a19107058c28c73633cc25b82a018fc279034d6670b45488022d5071ce0` |
| `align_expected_112x112_rgb8.bin` sha256 | `0b04d1c71bd97ee3ea42f01fde36cd36282ed6ba4a85843613597fa6f4dc45c4` |

The arithmetic is IEEE-754 `f64` throughout with no BLAS call on the hot path,
so a different numpy or Python is expected to reproduce these bytes; the row
records what was actually run, not what is required.
