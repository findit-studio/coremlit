"""The audio recipe refuses a converted program that holds a beta-less
``layer_norm`` (a Core ML fp32 CPU-road defect; see
``probes/layer_norm_cpu_probe.py``).

Converting needs the pinned venv, so nothing here converts. What is exercised
is the detector on planted operations, and that both recipe steps use it.

Run (stdlib ``unittest``; no model, no venv):

    python3 -m unittest discover -s coremlit/conversion/embeddinggemma2/tests -v
"""
import os
import sys
import unittest
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), "scripts")
sys.path.insert(0, _SCRIPTS)

import _eg2_common  # noqa: E402  (stdlib only at import time)


def op(op_type, name, **inputs):
    return SimpleNamespace(op_type=op_type, inputs=inputs, outputs=[SimpleNamespace(name=name)])


class BetalessLayerNorms(unittest.TestCase):
    def test_a_layer_norm_without_a_beta_is_named(self):
        count, missing = _eg2_common.betaless_layer_norms(
            [op("layer_norm", "var_160", gamma=1), op("layer_norm", "var_189", gamma=1, beta=0)]
        )
        self.assertEqual((count, missing), (2, ["var_160"]))

    def test_a_beta_mapped_to_none_counts_as_missing(self):
        count, missing = _eg2_common.betaless_layer_norms([op("layer_norm", "var_1", gamma=1, beta=None)])
        self.assertEqual((count, missing), (1, ["var_1"]))

    def test_explicit_betas_leave_nothing_missing(self):
        count, missing = _eg2_common.betaless_layer_norms(
            [op("layer_norm", "a", beta=0), op("add", "b"), op("layer_norm", "c", beta=0)]
        )
        self.assertEqual((count, missing), (2, []))

    def test_other_operations_are_not_counted(self):
        self.assertEqual(_eg2_common.betaless_layer_norms([op("add", "a"), op("pow", "b")]), (0, []))


class RefusalsAreByName(unittest.TestCase):
    def _with_operations(self, operations):
        original = _eg2_common.package_layer_norms
        _eg2_common.package_layer_norms = lambda path: _eg2_common.betaless_layer_norms(operations)
        self.addCleanup(setattr, _eg2_common, "package_layer_norms", original)

    def test_a_betaless_program_is_refused_naming_the_operations(self):
        self._with_operations([op("layer_norm", "var_160", gamma=1), op("layer_norm", "var_189", gamma=1)])
        with self.assertRaises(SystemExit) as caught:
            _eg2_common.require_explicit_beta("pkg")
        self.assertIn("2 of 2 layer_norm operations have no beta (var_160, var_189)", str(caught.exception))

    def test_a_program_with_no_layer_norm_is_refused_as_vacuous(self):
        self._with_operations([op("add", "a")])
        with self.assertRaises(SystemExit) as caught:
            _eg2_common.require_explicit_beta("pkg")
        self.assertIn("vacuously", str(caught.exception))

    def test_explicit_betas_pass_and_are_counted(self):
        self._with_operations([op("layer_norm", "a", beta=0), op("layer_norm", "b", beta=0)])
        self.assertEqual(_eg2_common.require_explicit_beta("pkg"), 2)


class TheRecipeUsesIt(unittest.TestCase):
    def _text(self, *parts):
        with open(os.path.join(os.path.dirname(_HERE), *parts), encoding="utf-8") as f:
            return f.read()

    def test_conversion_gives_a_zero_bias_and_checks_the_saved_package(self):
        text = self._text("scripts", "convert_audio_fp32.py")
        self.assertIn("give_layer_norms_a_zero_bias(self)", text)
        self.assertIn("require_explicit_beta(target)", text)

    def test_verification_checks_the_package_and_both_roads(self):
        text = self._text("scripts", "verify_audio_fp32.py")
        self.assertIn("require_explicit_beta(path)", text)
        self.assertIn("ct.ComputeUnit.CPU_ONLY", text)
        self.assertIn("ct.ComputeUnit.CPU_AND_GPU", text)

    def test_the_probe_ships_beside_the_recipe(self):
        self.assertTrue(os.path.isfile(os.path.join(os.path.dirname(_HERE), "probes", "layer_norm_cpu_probe.py")))


if __name__ == "__main__":
    unittest.main()
