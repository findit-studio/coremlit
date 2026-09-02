"""Price the WINDOW LENGTH decision, so it is a measurement rather than a default.

ReDimNet never downsamples time — the stage convolution is
``Conv2d(c, stride*c*conv_exp, kernel_size=(stride,1), stride=(stride,1))`` over a
``(B, C, F, T)`` tensor, i.e. FREQUENCY-only striding — so every stage runs at the mel
rate and cost is linear in T, EXCEPT the six 4-head self-attention blocks, which are
quadratic. This script measures both halves at several candidate windows:

  * wall clock, PyTorch fp32 eager (the shape of the curve, not a CoreML claim);
  * the analytic quadratic term, ``2 * T^2 * sum(hC)`` MACs over the six ``TimeContextBlock1d``
    attentions, where ``hC = (C*F) // att_block_red`` per stage — computed from the
    checkpoint's OWN ``stages_setup``, not from the paper;
  * the 1-D activation footprint, ``(num_stages + 2) * C * F * T`` values, because
    ``weigth1d`` keeps every previous stage output live to softmax-weight them, and
    activation traffic rather than MACs is usually what decides ANE residency.

Usage: ``python measure_window_cost.py [seconds …]`` (default: 2 3 6 8 10).
"""
import sys
import time

import numpy as np
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _redimnet_common import HOP_LENGTH, SAMPLE_RATE, WINDOW_SAMPLES, load_model

REPEATS = 5


def frames(n_samples):
    return 1 + n_samples // HOP_LENGTH


def attention_quadratic_macs(cfg, T):
    """QK^T and attn@V for each stage's 4-head attention: 2 * hC * T^2 each."""
    cf = cfg["C"] * cfg["F"]
    total = 0
    for _stride, _nb, _ce, _ks, att_block_red in cfg["stages_setup"]:
        if att_block_red is None:
            continue
        total += 2 * (cf // att_block_red) * T * T
    return total


def main(seconds):
    model, cfg = load_model()
    cf = cfg["C"] * cfg["F"]
    live_1d = len(cfg["stages_setup"]) + 2      # stem + one per stage + the weighted sum
    print(f"\nC*F = {cf}, stages = {len(cfg['stages_setup'])}, live 1-D tensors = {live_1d}")
    print("\n| window | samples | mel frames T | PyTorch fp32 warm (ms) | attention quadratic "
          "(MMAC) | 1-D activations fp16 (MiB) |")
    print("|---|---|---|---|---|---|")
    for s in seconds:
        n = int(round(s * SAMPLE_RATE))
        T = frames(n)
        x = torch.from_numpy(np.zeros((1, n), np.float32))
        with torch.no_grad():
            model(x)                                    # warm
            ts = []
            for _ in range(REPEATS):
                t0 = time.perf_counter()
                model(x)
                ts.append(1000.0 * (time.perf_counter() - t0))
        q = attention_quadratic_macs(cfg, T) / 1e6
        mib = live_1d * cf * T * 2 / (1024 ** 2)
        mark = "  <- chosen" if n == WINDOW_SAMPLES else ""
        print(f"| {s:g} s | {n} | {T} | {np.median(ts):.0f} | {q:.1f} | {mib:.1f} |{mark}")


if __name__ == "__main__":
    args = [float(a) for a in sys.argv[1:]] or [2, 3, 6, 8, 10]
    main(args)
