"""The four-arm placement sweep — the number the ReDimNet census line is waiting on.

For each of ``All``, ``CpuAndGpu``, ``CpuOnly`` and ``CpuAndNeuralEngine``: LOAD the fp16
``.mlmodelc``, PREDICT on every corpus clip, and record cold load, first predict, warm
predict, agreement with the fp32 CPU reference, and any ``BNNS Graph Shape Deduction``
line the runtime wrote to stderr.

The BNNS line is the tell. ``src/audio/lid`` records what it looks like when a graph does
not fit the ANE: ``BNNS Graph Shape Deduction: Unsupported kernel id 512`` alongside a
20× load and a 10× predict. That is the shape of a negative result, and a negative result
here is LOAD-BEARING for the whole ReDimNet family — B4 is the same graph as B5 (they
differ only in ``group_divisor``), and the 32 ``fwSE`` gates whose rank-4 -> rank-2 ->
rank-4 round trip is the suspected ANE-hostile op class exist in B4/B5 and in no other
size. So this one sweep answers B4 too, and rules on the op class the census flagged.

Arms run in separate processes (see ``_placement_arm.py``) so stderr is attributable and
each load is genuinely cold.

Usage: ``python sweep_placement.py [repeats]`` (default 10).
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import coremltools as ct

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _redimnet_common import (INPUT_NAME, OUTPUT_NAME, WINDOW_SAMPLES, cos, load_model,
                              mel_for_waveform, models_out_dir, staging_dir, worst_update)
from _fixtures import CORPUS, samples_f32

ARMS = ("All", "CpuAndGpu", "CpuOnly", "CpuAndNeuralEngine")
BNNS_RE = re.compile(r"BNNS Graph Shape Deduction[^\n]*")
# Wide sanity floor, MEASURED not marketed: true fp16 on the ANE still clears it.
SANITY_COS = 0.99


def main(repeats=10):
    bundle = models_out_dir() / "redimnet_b5.mlmodelc"
    if not bundle.is_dir():
        raise SystemExit(f"missing compiled bundle {bundle} — run the compile step first")

    clips = list(CORPUS)
    model, _cfg = load_model()
    xs = np.stack([mel_for_waveform(model, samples_f32(c, WINDOW_SAMPLES)[None, :]).numpy()
                   for c in clips]).astype(np.float32)
    inputs_npy = staging_dir() / "sweep_inputs.npy"
    np.save(inputs_npy, xs)

    # fp32 CPU reference — the same role CpuOnly plays in tests/*/placement.rs, but taken
    # from the fp32 graph so an fp16 arm cannot agree with a wrong reference.
    fp32 = ct.models.MLModel(str(staging_dir() / "redimnet_b5_fp32.mlpackage"),
                             compute_units=ct.ComputeUnit.CPU_ONLY)
    refs = [np.asarray(fp32.predict({INPUT_NAME: x})[OUTPUT_NAME], np.float64).ravel()
            for x in xs]

    rows = []
    for arm in ARMS:
        print(f"\n=== {arm} ===", flush=True)
        proc = subprocess.run(
            [sys.executable, "-u", str(Path(__file__).parent / "_placement_arm.py"),
             arm, str(bundle), str(inputs_npy), str(repeats)],
            capture_output=True, text=True)
        payload = None
        for line in proc.stdout.splitlines():
            if line.startswith("@@ARM@@"):
                payload = json.loads(line[len("@@ARM@@"):])
        bnns = sorted(set(BNNS_RE.findall(proc.stderr)))
        if payload is None:
            rows.append({"arm": arm, "error": f"child exited {proc.returncode} with no result",
                         "stderr_tail": proc.stderr[-800:], "bnns": bnns})
            print(f"  CHILD FAILED rc={proc.returncode}\n{proc.stderr[-800:]}")
            continue
        payload["bnns"] = bnns
        if payload["error"] is None:
            worst = 1.0
            for out, ref in zip(payload["embeddings"], refs):
                worst = worst_update(worst, cos(out, ref))
            payload["worst_cos_vs_fp32_cpu"] = worst
            payload["nan_free"] = bool(np.isfinite(np.asarray(payload["embeddings"])).all())
            del payload["embeddings"]
            print(f"  load {payload['load_ms']:8.1f} ms | first {payload['first_predict_ms']:7.1f} ms"
                  f" | warm {payload['warm_predict_ms']:7.1f} ms | worst cos {worst:.6f}"
                  f" | nan-free {payload['nan_free']}")
        else:
            print(f"  ERROR {payload['error']}")
        if bnns:
            for line in bnns:
                print(f"  stderr: {line}")
        else:
            print("  stderr: no BNNS Graph Shape Deduction lines")
        rows.append(payload)

    out = staging_dir() / "placement.json"
    out.write_text(json.dumps({"arms": rows, "repeats": repeats,
                               "clips": clips, "sanity_cos": SANITY_COS}, indent=2) + "\n")

    print("\n| arm | load (ms) | first predict (ms) | warm predict (ms) | worst cos vs fp32 CPU | BNNS |")
    print("|---|---|---|---|---|---|")
    failures = []
    for r in rows:
        if r.get("error"):
            print(f"| {r['arm']} | — | — | — | — | {'yes' if r.get('bnns') else 'no'} |  ERROR: {r['error']}")
            failures.append(f"{r['arm']}: {r['error']}")
            continue
        print(f"| {r['arm']} | {r['load_ms']:.0f} | {r['first_predict_ms']:.1f} | "
              f"{r['warm_predict_ms']:.1f} | {r['worst_cos_vs_fp32_cpu']:.6f} | "
              f"{'yes' if r['bnns'] else 'no'} |")
        if not r["nan_free"]:
            failures.append(f"{r['arm']}: non-finite output")
        if not (r["worst_cos_vs_fp32_cpu"] >= SANITY_COS):
            failures.append(f"{r['arm']}: worst cos {r['worst_cos_vs_fp32_cpu']:.6f} "
                            f"< {SANITY_COS}")
    print(f"\nwrote {out}")
    if failures:
        print("\nPLACEMENT SWEEP FAILURES:")
        for f in failures:
            print("  -", f)
        sys.exit(1)
    print("\nDONE placement — every arm loaded, predicted, and agreed with the fp32 reference.")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10)
