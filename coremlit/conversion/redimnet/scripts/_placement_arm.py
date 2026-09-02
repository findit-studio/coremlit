"""ONE compute-unit arm of the placement sweep, in its OWN process.

Why a subprocess per arm rather than a loop. Two reasons, both load-bearing:

  * **stderr.** ``BNNS Graph Shape Deduction: …`` is written by the CoreML/ANE runtime at
    the C++ layer, straight to file descriptor 2. Nothing a Python ``contextlib`` context
    manager does to ``sys.stderr`` can see it. A child process whose fd 2 the parent owns
    can, and the parent can attribute every line to exactly one arm.
  * **cold load.** Load time is the number that exposed the pathological ANE arm in
    ``src/audio/lid`` (2 440 ms against 113 ms). CoreML caches compiled ANE programs per
    process, so the second load in one process is not a load at all. A fresh process per
    arm is the only way each arm's load time means the same thing.

Usage: ``python _placement_arm.py <arm> <mlmodelc> <inputs.npy> <repeats>``. The result is
one line on stdout prefixed ``@@ARM@@`` (everything else on stdout is noise from the
runtime and is ignored by the parent).
"""
import json
import sys
import time

import numpy as np
import coremltools as ct

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _redimnet_common import INPUT_NAME, OUTPUT_NAME

UNITS = {
    "All": ct.ComputeUnit.ALL,
    "CpuAndGpu": ct.ComputeUnit.CPU_AND_GPU,
    "CpuOnly": ct.ComputeUnit.CPU_ONLY,
    "CpuAndNeuralEngine": ct.ComputeUnit.CPU_AND_NE,
}


def main(arm, path, inputs_npy, repeats):
    xs = np.load(inputs_npy)                       # [n_clips, 1, WINDOW_SAMPLES] f32
    result = {"arm": arm, "load_ms": None, "first_predict_ms": None,
              "warm_predict_ms": None, "embeddings": None, "error": None}
    try:
        t0 = time.perf_counter()
        model = ct.models.CompiledMLModel(path, UNITS[arm])
        result["load_ms"] = 1000.0 * (time.perf_counter() - t0)

        outs = []
        t0 = time.perf_counter()
        first = model.predict({INPUT_NAME: xs[0]})[OUTPUT_NAME]
        result["first_predict_ms"] = 1000.0 * (time.perf_counter() - t0)
        outs.append(np.asarray(first, np.float64).ravel().tolist())

        for x in xs[1:]:
            outs.append(np.asarray(model.predict({INPUT_NAME: x})[OUTPUT_NAME],
                                   np.float64).ravel().tolist())
        # Warm latency: median over `repeats` runs of one clip, after the graph is hot.
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            model.predict({INPUT_NAME: xs[0]})
            times.append(1000.0 * (time.perf_counter() - t0))
        result["warm_predict_ms"] = float(np.median(times))
        result["embeddings"] = outs
    except Exception as exc:                        # a failed arm is a RESULT, not a crash
        result["error"] = f"{type(exc).__name__}: {exc}"
    print("@@ARM@@" + json.dumps(result))


if __name__ == "__main__":
    if len(sys.argv) != 5:
        raise SystemExit("usage: _placement_arm.py <arm> <mlmodelc> <inputs.npy> <repeats>")
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]))
