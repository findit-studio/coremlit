"""The four-arm placement sweep, plus the throughput number issue #115's Acceptance asks
for. Usage: ``python sweep_placement.py [repeats] [rounds]`` (default 100 predicts per arm
per round, 5 rounds).

For each of ``All``, ``CpuAndGpu``, ``CpuOnly`` and ``CpuAndNeuralEngine``: LOAD the fp16
``.mlmodelc`` in a fresh process, PREDICT on every fixture face, and record cold load, first
predict, warm predict, agreement with the fp32 CoreML reference, and any
``BNNS Graph Shape Deduction`` line the runtime wrote to stderr.

**Why rounds, and not one pass.** The first two single-pass runs of this sweep disagreed
about the winner — ``CpuAndNeuralEngine`` 4.38 ms against ``All`` 9.91 ms in one, then 4.90
against 4.84 in the next. A warm predict of a few milliseconds on a shared desktop is within
reach of whatever else the machine is doing, and a recommendation taken from one draw is a
recommendation about the machine's mood. Five rounds put it beyond doubt (ANE 3.14–4.36 ms,
``All`` 4.63–8.70), so the sweep reports the MEDIAN ACROSS ROUNDS and the per-round spread
beside it, and recommends on the median.

The BNNS line is the other tell. ``src/audio/lid`` records what it looks like when a graph
does not fit the Neural Engine — ``BNNS Graph Shape Deduction: Unsupported kernel id 512``
alongside a 20x load and a 10x predict — so a clean stderr plus a fast warm predict is what
a resident graph looks like, and neither alone is evidence.

**The recommendation is the measurement's, not a default.** ReDimNet-B5's sweep chose
``CpuAndGpu``; nothing about that transfers.

**Throughput.** The artifact's batch is 1, so a keyframe with N faces is N predicts through
one loaded model, and ``faces/s`` is reported that way — warm, on the recommended arm —
because that is what the door will actually do. A batch-8 export would change this number
and is the follow-up issue #115's census sized at ~1.7x.
"""
import json
import re
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from _arcface_common import (BUNDLE, INPUT_NAME, MLPACKAGE_FP32, OUTPUT_NAME, conv_dir, cos,
                             fixtures_dir, models_out_dir, observed_compiler,
                             observed_toolchain, preprocess, staging_dir)

ARMS = ("All", "CpuAndGpu", "CpuOnly", "CpuAndNeuralEngine")
BNNS_RE = re.compile(r"BNNS Graph Shape Deduction[^\n]*")
#: A wide sanity floor, MEASURED not marketed: true fp16 on the ANE still clears it. The
#: real parity gate is ``verify_arcface.py``'s; this one only catches an arm computing
#: something else entirely.
SANITY_COS = 0.99


def fixture_batch():
    manifest = json.loads((fixtures_dir() / "faces" / "manifest.json").read_text())
    crops = np.stack([
        np.frombuffer((fixtures_dir() / "faces" / row["crop"]).read_bytes(),
                      np.uint8).reshape(112, 112, 3)
        for row in manifest["faces"]])
    return preprocess(crops)[:, None, ...]          # [n, 1, 3, 112, 112]


def run_arm(arm, bundle, inputs_npy, repeats):
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
        return {"arm": arm, "error": f"child exited {proc.returncode} with no result",
                "stderr_tail": proc.stderr[-800:], "bnns": bnns}
    payload["bnns"] = bnns
    return payload


def main(repeats=100, rounds=5):
    observed = observed_toolchain()
    bundle = models_out_dir() / BUNDLE
    if not bundle.is_dir():
        raise SystemExit(f"missing compiled bundle {bundle} — run the compile step first")

    xs = fixture_batch()
    inputs_npy = staging_dir() / "sweep_inputs.npy"
    np.save(inputs_npy, xs)

    # The fp32 CoreML graph is the reference, not one of the fp16 arms: an fp16 arm cannot
    # be scored against another fp16 arm without the two agreeing on a shared error.
    import coremltools as ct
    fp32 = ct.models.MLModel(str(staging_dir() / MLPACKAGE_FP32),
                             compute_units=ct.ComputeUnit.CPU_ONLY)
    refs = [np.asarray(fp32.predict({INPUT_NAME: x})[OUTPUT_NAME], np.float64).ravel()
            for x in xs]

    draws = {arm: [] for arm in ARMS}
    for round_index in range(rounds):
        print(f"\n--- round {round_index + 1}/{rounds} ---", flush=True)
        for arm in ARMS:
            payload = run_arm(arm, bundle, inputs_npy, repeats)
            if payload.get("error") is None:
                embeddings = payload.pop("embeddings")
                cs = [cos(r, e) for r, e in zip(refs, embeddings)]
                payload["cos_vs_fp32_min"] = min(cs)
                if min(cs) < SANITY_COS:
                    payload["error"] = f"agreement with fp32 CoreML {min(cs)} < {SANITY_COS}"
                print(f"  {arm:20s} load {payload['load_ms']:7.1f}  first "
                      f"{payload['first_predict_ms']:7.1f}  warm {payload['warm_predict_ms']:6.2f} ms"
                      f"  cos {min(cs):.6f}  {'; '.join(payload['bnns']) or 'BNNS clean'}")
            else:
                payload.pop("embeddings", None)
                print(f"  {arm:20s} ERROR {payload['error']}")
            draws[arm].append(payload)

    rows = []
    for arm in ARMS:
        good = [d for d in draws[arm] if not d.get("error")]
        if not good:
            rows.append({"arm": arm, "error": draws[arm][0].get("error"), "rounds": rounds,
                         "bnns": sorted({b for d in draws[arm] for b in d.get("bnns", [])})})
            continue
        warm = [d["warm_predict_ms"] for d in good]
        rows.append({
            "arm": arm, "rounds": len(good),
            "load_ms_median": statistics.median(d["load_ms"] for d in good),
            "first_predict_ms_median": statistics.median(d["first_predict_ms"] for d in good),
            "warm_predict_ms_median": statistics.median(warm),
            "warm_predict_ms_min": min(warm), "warm_predict_ms_max": max(warm),
            "warm_predict_ms_per_round": warm,
            "faces_per_second": 1000.0 / statistics.median(warm),
            "cos_vs_fp32_min": min(d["cos_vs_fp32_min"] for d in good),
            "bnns": sorted({b for d in good for b in d["bnns"]}),
            "error": None,
        })

    ok = [r for r in rows if not r.get("error")]
    recommended = min(ok, key=lambda r: r["warm_predict_ms_median"]) if ok else None

    print(f"\n| arm | cold load ms | first predict ms | warm predict ms "
          f"(median of {rounds}) | faces/s | min cos vs fp32 | BNNS |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        if r.get("error"):
            print(f"| {r['arm']} | — | — | — | — | — | {r['error']} |")
            continue
        print(f"| {r['arm']} | {r['load_ms_median']:.0f} | {r['first_predict_ms_median']:.0f} | "
              f"{r['warm_predict_ms_median']:.2f} ({r['warm_predict_ms_min']:.2f}–"
              f"{r['warm_predict_ms_max']:.2f}) | {r['faces_per_second']:.0f} | "
              f"{r['cos_vs_fp32_min']:.6f} | {'; '.join(r['bnns']) or 'clean'} |")
    if recommended:
        print(f"\nRECOMMENDED: {recommended['arm']} "
              f"({recommended['warm_predict_ms_median']:.2f} ms warm, "
              f"{recommended['faces_per_second']:.0f} faces/s, cold load "
              f"{recommended['load_ms_median']:.0f} ms)")

    report = {
        "repeats_per_round": repeats, "rounds": rounds, "faces": int(xs.shape[0]),
        "arms": rows, "recommended": recommended["arm"] if recommended else None,
        "throughput_faces_per_second": recommended["faces_per_second"] if recommended else None,
        "method": (f"one loaded model in a fresh process per arm per round, one predict per "
                   f"face (the artifact's batch is 1); warm = median of {repeats} "
                   f"back-to-back predicts after the graph is hot, then the MEDIAN of "
                   f"{rounds} such rounds; faces/s = 1000 / warm_ms"),
        "toolchain": observed, "compiler": observed_compiler(),
    }
    (conv_dir() / "placement.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"[ok] wrote {conv_dir() / 'placement.json'}")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100,
         int(sys.argv[2]) if len(sys.argv) > 2 else 5)
