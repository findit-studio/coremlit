"""Write the token-table gather golden, ``gather_f16.bin``, and its provenance.

For every one of the 65,536 bf16 bit patterns ``b``, the golden holds the f16
bits of ``f16(f32(b << 16) * f32(sqrt(512)))``: the f32 product rounded to f16
by numpy, round to nearest even, subnormals kept. That is the gather the kit
performs on every token-table cell, computed here by an independent
implementation, so a conversion defect the kit's vector and scalar paths shared
could not hide from it.

Little-endian u16, in pattern order: 131,072 bytes. Usage:

    python gather_golden.py <out_dir>
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from _eg2_common import EMBED_SCALE_BITS, observed_toolchain, sha256_bytes  # noqa: E402

GOLDEN = "gather_f16.bin"
PROVENANCE = "gather_f16.provenance.json"


def gather_golden():
    patterns = np.arange(65_536, dtype=np.uint32)
    widened = (patterns << np.uint32(16)).view(np.float32)
    scale = np.array([EMBED_SCALE_BITS], dtype=np.uint32).view(np.float32)[0]
    with np.errstate(over="ignore", invalid="ignore"):
        product = widened * scale
        rounded = product.astype(np.float16)
    return widened, rounded


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    out_dir = sys.argv[1]
    toolchain = observed_toolchain()
    widened, rounded = gather_golden()
    data = rounded.view(np.uint16).astype("<u2").tobytes()
    assert len(data) == 131_072

    finite_in = np.isfinite(widened)
    finite_out = np.isfinite(rounded)
    counts = {
        "patterns": 65_536,
        "finite_inputs": int(finite_in.sum()),
        "nan_inputs": int(np.isnan(widened).sum()),
        "infinite_inputs": int(np.isinf(widened).sum()),
        "finite_inputs_overflowing_to_infinity": int((finite_in & np.isinf(rounded)).sum()),
        "finite_nonzero_inputs_rounding_to_zero": int((finite_in & (widened != 0) & (rounded == 0)).sum()),
        "subnormal_results": int((finite_out & (rounded != 0) & (np.abs(rounded) < np.float16(6.103515625e-05))).sum()),
    }
    # The design's scan: 65,280 finite patterns, of which 29,846 overflow, and
    # 254 NaN patterns. The golden must agree before it is written.
    assert counts["finite_inputs"] == 65_280, counts
    assert counts["finite_inputs_overflowing_to_infinity"] == 29_846, counts
    assert counts["nan_inputs"] == 254, counts

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, GOLDEN), "wb") as f:
        f.write(data)
    record = {
        "file": GOLDEN,
        "sha256": sha256_bytes(data),
        "layout": "65,536 little-endian u16, one per bf16 pattern in pattern order",
        "rule": "f16(f32(pattern << 16) * f32(0x41B504F3)), numpy float32 multiply then astype(float16): round to nearest even, subnormals kept",
        "nan_payloads": "not pinned: a NaN pattern maps to some NaN",
        "counts": counts,
        "toolchain": {"python": toolchain["python"], "numpy": toolchain["numpy"]},
        "generator": "coremlit/conversion/embeddinggemma2/scripts/gather_golden.py",
    }
    with open(os.path.join(out_dir, PROVENANCE), "w") as f:
        json.dump(record, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"[ok] {GOLDEN}: {record['sha256']}")
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
