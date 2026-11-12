"""Regression probe for a Core ML defect the audio recipe works around.

Core ML's fp32 CPU road (``CPU_ONLY``, and ``CPU_AND_GPU`` on a host with no
usable GPU, such as the CI runners) computes a ``layer_norm`` that has NO
``beta`` wrongly when it sits in this chain, which is the audio tower's
subsampling block:

    conv (3x3, stride 2) -> transpose -> layer_norm (128 channels, gamma) ->
    transpose -> relu

The result carries a stray per-channel offset on channels 64 and up (the
upper half of the 128), as if the kernel read a bias the graph does not
supply. The GPU road is right. A ``layer_norm`` alone is right on both roads
for 32, 64, 128 and 256 channels, so the defect belongs to the chain, not to
the operation. Seen on macOS 27.0 (26A428, Apple M1 Max) and on macOS 15.8.2
(24H232, the GitHub ``macos-15`` runner), so it is the framework's, not the
host's.

An explicit zero ``beta`` is the same function and removes it. The recipe
therefore gives every beta-less ``nn.LayerNorm`` a zero bias and refuses a
converted program that still holds a beta-less ``layer_norm``
(``_eg2_common.require_explicit_beta``).

This probe builds the chain both ways and compares the CPU road against the
GPU road:

- the chain WITH a zero ``beta`` must agree on both roads (a failure here exits
  non-zero: the workaround stopped working);
- the chain WITHOUT one is reported as ``defect present`` or ``defect absent``
  and exits zero either way, so a Core ML release that fixes the defect is
  visible rather than a failure.

Needs the producer's venv (coremltools 9.0); it is not run by CI.

    <venv>/bin/python coremlit/conversion/embeddinggemma2/probes/layer_norm_cpu_probe.py
"""
import sys
import tempfile

import numpy as np

AGREE = 1e-4
CHANNELS = 128


def build(with_beta, rng):
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb

    weight = (rng.standard_normal((CHANNELS, 1, 3, 3)) * 0.5).astype(np.float32)
    gamma = rng.uniform(-2, 8, CHANNELS).astype(np.float32)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 1, 1000, CHANNELS))], opset_version=ct.target.iOS16)
    def program(x):
        h = mb.conv(x=x, weight=weight, strides=[2, 2], pad_type="custom", pad=[1, 1, 1, 1])
        t = mb.transpose(x=h, perm=[0, 2, 3, 1])
        arguments = dict(x=t, axes=[-1], gamma=gamma, epsilon=np.float32(1e-6))
        if with_beta:
            arguments["beta"] = np.zeros(CHANNELS, np.float32)
        norm = mb.layer_norm(**arguments)
        return mb.relu(x=mb.transpose(x=norm, perm=[0, 3, 1, 2]))

    return ct.convert(
        program,
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.iOS16,
    )


def worst_difference(model, x):
    import coremltools as ct

    with tempfile.TemporaryDirectory() as directory:
        path = f"{directory}/probe.mlpackage"
        model.save(path)
        outputs = {}
        for name in ("CPU_ONLY", "CPU_AND_GPU"):
            loaded = ct.models.MLModel(path, compute_units=getattr(ct.ComputeUnit, name))
            feed = {loaded.get_spec().description.input[0].name: x}
            outputs[name] = next(iter(loaded.predict(feed).values()))
    return float(np.abs(outputs["CPU_ONLY"] - outputs["CPU_AND_GPU"]).max())


def main():
    rng = np.random.default_rng(0)
    x = (rng.standard_normal((1, 1, 1000, CHANNELS)) * 2).astype(np.float32)
    without = worst_difference(build(False, rng), x)
    with_zero = worst_difference(build(True, rng), x)
    print(f"no beta   : CPU against GPU, worst difference {without:.3e} -> defect {'present' if without > AGREE else 'absent'}")
    print(f"zero beta : CPU against GPU, worst difference {with_zero:.3e}")
    if with_zero > AGREE:
        sys.exit(f"the zero-beta workaround no longer makes the CPU road agree with the GPU road (> {AGREE})")


if __name__ == "__main__":
    main()
