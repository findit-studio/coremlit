"""The audio recipe's two cosine gates fail closed.

``convert_audio_fp32.py``'s faithfulness gate and ``verify_audio_fp32.py``'s
token-level parity each reduce a window to its worst token cosine and refuse
the window under a floor with ``worst < floor``. A NaN compares false against
every floor, so a NaN in an output, or a token of zero norm, used to pass both
gates and let a broken package be staged. Both gates now route through
``_eg2_common.worst_token_cosine``, which refuses a cosine that is not a finite
number by name, before any floor is compared.

WHAT THESE TESTS DO NOT COVER. Running the conversion or the verification
needs the pinned venv and the checkpoint, so nothing here traces, converts or
predicts. What is exercised is the helper itself, on planted outputs, and that
both gates call it.

Run (stdlib ``unittest`` and numpy; no model, no venv):

    python3 -m unittest discover -s coremlit/conversion/embeddinggemma2/tests -v
"""
import os
import sys
import unittest

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS = os.path.join(os.path.dirname(_HERE), "scripts")
sys.path.insert(0, _SCRIPTS)

import _eg2_common  # noqa: E402  (stdlib only at import time)


def _tokens(rows=4, width=8):
    """Two identical windows of ``rows`` tokens: every cosine is exactly 1."""
    ours = np.linspace(0.5, 1.5, rows * width, dtype=np.float32).reshape(rows, width)
    return ours, ours.copy()


class WorstTokenCosine(unittest.TestCase):
    def test_a_planted_nan_output_is_refused_by_name(self):
        ours, theirs = _tokens()
        ours[2, 5] = np.nan
        with self.assertRaises(SystemExit) as caught:
            _eg2_common.worst_token_cosine(ours, theirs, "jfk@0")
        message = str(caught.exception)
        self.assertIn("jfk@0: token 2", message)
        self.assertIn("not a finite number", message)

    def test_a_zero_norm_token_is_refused_by_name(self):
        ours, theirs = _tokens()
        ours[1] = 0.0
        with self.assertRaises(SystemExit) as caught:
            _eg2_common.worst_token_cosine(ours, theirs, "faithfulness: jfk 1 s tail")
        self.assertIn("faithfulness: jfk 1 s tail: token 1", str(caught.exception))

    def test_an_infinite_output_is_refused_by_name(self):
        ours, theirs = _tokens()
        theirs[3, 0] = np.inf
        with self.assertRaises(SystemExit) as caught:
            _eg2_common.worst_token_cosine(ours, theirs, "sine440@0")
        self.assertIn("sine440@0: token 3", str(caught.exception))

    def test_an_empty_window_is_refused(self):
        with self.assertRaises(SystemExit) as caught:
            _eg2_common.worst_token_cosine(np.zeros((0, 8)), np.zeros((0, 8)), "empty")
        self.assertIn("empty: no token to compare", str(caught.exception))

    def test_finite_tokens_give_their_worst_cosine(self):
        ours, theirs = _tokens()
        theirs[0] = -theirs[0]
        self.assertAlmostEqual(_eg2_common.worst_token_cosine(ours, theirs, "w"), -1.0, places=12)
        self.assertAlmostEqual(_eg2_common.worst_token_cosine(ours, ours, "w"), 1.0, places=12)


class BothGatesRouteThroughIt(unittest.TestCase):
    def test_both_gates_call_the_fail_closed_helper(self):
        for script in ("convert_audio_fp32.py", "verify_audio_fp32.py"):
            with open(os.path.join(_SCRIPTS, script), encoding="utf-8") as f:
                text = f.read()
            self.assertTrue("worst_token_cosine(" in text, f"{script} does not call worst_token_cosine")
            self.assertFalse("cosines(" in text, f"{script} keeps a bare cosine reduction")


if __name__ == "__main__":
    unittest.main()
