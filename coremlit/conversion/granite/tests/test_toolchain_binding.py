"""The producing toolchain must be the SAME one every later stage runs under.

``REQUIRED_TOOLCHAIN`` pins python to MAJOR.MINOR, so 3.11.14 and 3.11.15 both
satisfy ``observed_toolchain()``. That is deliberate — and it is why the pins
alone cannot answer "was this artifact made by one environment?". Conversion
under 3.11.14, goldens under 3.11.15 and verification back under 3.11.14 used to
publish cleanly: the conversion-time crosscheck and the embeddings it is
published beside were computed in different environments, while every id, digest
and run-identity check still held.

``verify_granite.py`` has always closed its half. These tests cover the goldens
half and the shared guard both stages now route through.

WHAT THESE TESTS DO NOT COVER. Running the real conversion is owner-gated (it
needs the pinned venv and the checkpoint), so nothing here computes an
embedding, traces a graph or writes a golden. What is exercised is
``main()``'s control flow up to the model load, with the heavy dependencies
absent and ``observed_toolchain()`` supplied by the test. That the guard fires
BEFORE ``load_sentence_transformer()`` is itself part of the contract asserted
here: a divergence must cost nothing to discover, and moving the guard after the
model load would red these tests.

Run:

    python3 -m unittest discover -s coremlit/conversion/granite/tests -v
"""
import json
import os
import sys
import types
import unittest
from unittest import mock

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), "scripts")


def _stub(name, build):
    """Install a minimal stand-in for ``name`` only when it is not installed.

    The recipe's modules import torch and coremltools at module scope, but every
    code path under test here is dict comparison and file reading. Stubbing lets
    the guard be tested in any interpreter; inside the pinned venv the real
    modules are found first and nothing is replaced.
    """
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = build()


def _torch_stub():
    mod = types.ModuleType("torch")
    nn = types.ModuleType("torch.nn")
    # `class GraniteGraph(torch.nn.Module)` is the only torch use evaluated at
    # import time in _granite_common.
    nn.Module = type("Module", (), {})
    mod.nn = nn
    sys.modules["torch.nn"] = nn
    return mod


def _coremltools_stub():
    mod = types.ModuleType("coremltools")
    # verify_granite builds its compute-unit table at module scope.
    mod.ComputeUnit = type("ComputeUnit", (), {
        "CPU_ONLY": "CPU_ONLY",
        "CPU_AND_GPU": "CPU_AND_GPU",
        "CPU_AND_NE": "CPU_AND_NE",
        "ALL": "ALL",
    })
    return mod


_stub("torch", _torch_stub)
_stub("coremltools", _coremltools_stub)

sys.path.insert(0, _SCRIPTS)

import _granite_common as gc          # noqa: E402
import generate_goldens               # noqa: E402
import verify_granite                 # noqa: E402

# Two readings that BOTH satisfy REQUIRED_TOOLCHAIN — the divergence the pins
# cannot see. Only the python patch level differs, which is the narrowest
# mismatch the guard has to catch.
PRODUCER_TOOLCHAIN = {
    "python": "3.11.14",
    "torch": "2.6.0",
    "transformers": "5.14.0",
    "sentence_transformers": "5.6.0",
    "coremltools": "9.0",
    "numpy": "1.26.4",
}
GENERATOR_TOOLCHAIN = dict(PRODUCER_TOOLCHAIN, python="3.11.15")


class _ReachedModelLoad(Exception):
    """Raised by the stand-in loader: control flow got past the guard."""


class _StagingFixture:
    """A staging tree carrying two packages and a producer record over them.

    Only the shapes the toolchain guard's call sites read are built: the
    ``.mlpackage`` directories ``read_producer_record`` is bound to, and the
    record itself. No CoreML data is involved.
    """

    def __init__(self, tmp, toolchain):
        self.stage = os.path.join(tmp, "staging")
        self.goldens = os.path.join(tmp, "goldens")
        self.models_out = os.path.join(tmp, "models")
        os.makedirs(self.stage)
        os.makedirs(self.goldens)
        os.makedirs(self.models_out)
        for name in (gc.SHIPPED_PACKAGE, gc.FP32_REFERENCE):
            pkg = os.path.join(self.stage, name)
            os.makedirs(pkg)
            with open(os.path.join(pkg, "model.mil"), "w") as f:
                f.write(name)
        record = {
            gc.RUN_ID_KEY: "0123456789abcdef0123456789abcdef",
            "converted_utc": "2026-07-26T14:00:00+00:00",
            "produced": {
                name: gc.digest_tree(os.path.join(self.stage, name))
                for name in (gc.SHIPPED_PACKAGE, gc.FP32_REFERENCE)
            },
        }
        if toolchain is not None:
            record["toolchain"] = toolchain
        with open(os.path.join(self.stage, gc.PRODUCER_RECORD), "w") as f:
            json.dump(record, f, indent=2)

    def env(self):
        return {
            "GRANITE_STAGE": self.stage,
            "GRANITE_GOLDENS": self.goldens,
            "GRANITE_MODELS_OUT": self.models_out,
            "GRANITE_CONV": os.path.dirname(self.stage),
        }


class ToolchainGuardTest(unittest.TestCase):
    """``require_producer_toolchain`` — the one comparison both stages call."""

    def test_identical_readings_pass(self):
        gc.require_producer_toolchain(
            {"toolchain": dict(PRODUCER_TOOLCHAIN)},
            dict(PRODUCER_TOOLCHAIN), "GOLDENS", "generate_goldens.py")

    def test_key_order_is_not_a_divergence(self):
        reordered = {k: PRODUCER_TOOLCHAIN[k] for k in reversed(list(PRODUCER_TOOLCHAIN))}
        gc.require_producer_toolchain(
            {"toolchain": reordered},
            dict(PRODUCER_TOOLCHAIN), "GOLDENS", "generate_goldens.py")

    def test_message_names_the_diverging_field_and_both_values(self):
        with self.assertRaises(SystemExit) as caught:
            gc.require_producer_toolchain(
                {"toolchain": dict(PRODUCER_TOOLCHAIN)},
                dict(GENERATOR_TOOLCHAIN), "GOLDENS", "generate_goldens.py")
        msg = str(caught.exception)
        self.assertIn("python", msg)
        self.assertIn("3.11.14", msg)
        self.assertIn("3.11.15", msg)
        self.assertIn("generate_goldens.py", msg)
        # Only the field that actually moved is reported, so the operator is not
        # asked to diff two six-entry dicts by eye.
        for quiet in ("torch", "transformers", "sentence_transformers", "numpy"):
            self.assertNotIn(f"{quiet}:", msg)

    def test_extra_producer_field_diverges(self):
        with self.assertRaises(SystemExit) as caught:
            gc.require_producer_toolchain(
                {"toolchain": dict(PRODUCER_TOOLCHAIN, scipy="1.0.0")},
                dict(PRODUCER_TOOLCHAIN), "GOLDENS", "generate_goldens.py")
        self.assertIn("scipy", str(caught.exception))

    def test_absent_record_is_rejected_not_waved_through(self):
        for record in ({}, {"toolchain": None}, {"toolchain": "3.11.14"}):
            with self.subTest(record=record):
                with self.assertRaises(SystemExit) as caught:
                    gc.require_producer_toolchain(
                        record, dict(PRODUCER_TOOLCHAIN), "GOLDENS", "generate_goldens.py")
                self.assertIn("convert_granite.py", str(caught.exception))


class GenerateGoldensTest(unittest.TestCase):
    """The goldens stage must refuse a producer it did not share an env with."""

    def _run_main(self, toolchain, observed):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _StagingFixture(tmp, toolchain)
            with mock.patch.dict(os.environ, fixture.env()), \
                 mock.patch.object(generate_goldens, "observed_toolchain",
                                   lambda: dict(observed)), \
                 mock.patch.object(generate_goldens, "load_sentence_transformer",
                                   mock.Mock(side_effect=_ReachedModelLoad())):
                generate_goldens.main()

    def test_mismatch_reds(self):
        with self.assertRaises(SystemExit) as caught:
            self._run_main(PRODUCER_TOOLCHAIN, GENERATOR_TOOLCHAIN)
        msg = str(caught.exception)
        self.assertIn("TOOLCHAIN DIVERGENCE", msg)
        self.assertIn("3.11.14", msg)
        self.assertIn("3.11.15", msg)

    def test_match_proceeds_past_the_guard(self):
        # The positive control. Reaching the loader proves the guard passed
        # rather than that it is absent — the mismatch case above fails first.
        with self.assertRaises(_ReachedModelLoad):
            self._run_main(PRODUCER_TOOLCHAIN, PRODUCER_TOOLCHAIN)

    def test_record_without_a_toolchain_reds(self):
        # A record predating the field cannot be shown to describe this
        # environment, so it is refused rather than waved through. Which layer
        # refuses it is not the contract — read_producer_record gets there first
        # — but it must name the missing field and demand a re-run.
        with self.assertRaises(SystemExit) as caught:
            self._run_main(None, PRODUCER_TOOLCHAIN)
        msg = str(caught.exception)
        self.assertIn("toolchain", msg)
        self.assertIn("re-run", msg.lower())


class VerifyGraniteTest(unittest.TestCase):
    """The verifier's half of the same guard, kept covered through the refactor."""

    def _run_main(self, toolchain, observed):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            fixture = _StagingFixture(tmp, toolchain)
            with mock.patch.dict(os.environ, fixture.env()), \
                 mock.patch.object(verify_granite, "observed_toolchain",
                                   lambda: dict(observed)), \
                 mock.patch.object(verify_granite, "require_compile_record",
                                   mock.Mock(side_effect=_ReachedModelLoad())):
                verify_granite.main()

    def test_mismatch_reds(self):
        with self.assertRaises(SystemExit) as caught:
            self._run_main(PRODUCER_TOOLCHAIN, GENERATOR_TOOLCHAIN)
        msg = str(caught.exception)
        self.assertIn("TOOLCHAIN DIVERGENCE", msg)
        self.assertIn("3.11.14", msg)
        self.assertIn("3.11.15", msg)

    def test_match_proceeds_past_the_guard(self):
        with self.assertRaises(_ReachedModelLoad):
            self._run_main(PRODUCER_TOOLCHAIN, PRODUCER_TOOLCHAIN)


if __name__ == "__main__":
    unittest.main()
