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


def current_report():
    windows = [
        {"clip": c, "start": 0, "road": r, "worst_cosine": 0.99999}
        for c in ("jfk", "sine440")
        for r in ("CPU_ONLY", "CPU_AND_GPU")
    ]
    return {
        "passed": True,
        "compute_units": ["CPU_ONLY", "CPU_AND_GPU"],
        "layer_norm_operations": 2,
        "windows": windows,
    }


class StagingRefusesAStaleVerification(unittest.TestCase):
    def refused(self, report):
        with self.assertRaises(SystemExit) as caught:
            _eg2_common.require_current_verification(report)
        return str(caught.exception)

    def test_a_current_report_is_accepted(self):
        _eg2_common.require_current_verification(current_report())

    def test_a_gpu_only_report_from_the_first_recipe_is_refused(self):
        legacy = current_report()
        legacy["compute_units"] = "CPU_AND_GPU"
        del legacy["layer_norm_operations"]
        for w in legacy["windows"]:
            del w["road"]
        self.assertIn("older verification", self.refused(legacy))

    def test_a_report_with_no_layer_norm_check_is_refused(self):
        report = current_report()
        report["layer_norm_operations"] = 0
        self.assertIn("no layer_norm check", self.refused(report))

    def test_a_window_judged_on_one_road_only_is_named(self):
        report = current_report()
        report["windows"] = [w for w in report["windows"] if not (w["clip"] == "jfk" and w["road"] == "CPU_ONLY")]
        self.assertIn("one road only: ('jfk', 0)", self.refused(report))

    def test_a_report_with_no_cpu_window_is_refused(self):
        report = current_report()
        report["windows"] = [w for w in report["windows"] if w["road"] != "CPU_ONLY"]
        self.assertIn("no window judged on CPU_ONLY", self.refused(report))

    def test_a_report_that_does_not_record_a_pass_is_refused(self):
        report = current_report()
        report["passed"] = False
        self.assertIn("does not record a pass", self.refused(report))


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
        self.assertIn("for name in VERIFY_ROADS", text)
        self.assertEqual(_eg2_common.VERIFY_ROADS, ("CPU_ONLY", "CPU_AND_GPU"))

    def test_staging_checks_the_report_and_verification_clears_a_stale_one(self):
        self.assertIn("require_current_verification(verify)", self._text("scripts", "stage_manifest.py"))
        verify = self._text("scripts", "verify_audio_fp32.py")
        self.assertIn("os.unlink(os.path.join(stage, VERIFY_FILE))", verify)
        self.assertIn("VERIFY_ROADS", verify)

    def test_the_probe_ships_beside_the_recipe(self):
        self.assertTrue(os.path.isfile(os.path.join(os.path.dirname(_HERE), "probes", "layer_norm_cpu_probe.py")))


if __name__ == "__main__":
    unittest.main()
